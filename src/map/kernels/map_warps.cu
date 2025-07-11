
#pragma once

namespace GpuBTree {
namespace warps {
#define KEY_PIVOT_MASK 0xAAAAAAA8
#define PIVOT_KEY_MASK 0x55555554

__forceinline__ __device__ uint32_t volatileRead(uint32_t* address) {
  uint32_t data;
#if __CUDA_ARCH__ >= 700
  asm volatile("ld.global.relaxed.sys.u32 %0, [%1];" : "=r"(data) : "l"(address));
#else
  data = *address;
#endif

  return data;
}
__forceinline__ __device__ void volatileWrite(uint32_t* address, uint32_t data) {
#if __CUDA_ARCH__ >= 700
  asm volatile("st.global.relaxed.sys.u32 [%0], %1;" ::"l"(address), "r"(data));
#else
  *address = data;
#endif
}

__forceinline__ __device__ uint32_t volatileNodeReadR(uint32_t* nodeAddress) {
  return volatileRead(nodeAddress + LANEID_REVERSED(lane_id()));
}

__forceinline__ __device__ uint32_t volatileNodeRead(uint32_t* nodeAddress) {
  return volatileRead(nodeAddress + lane_id());
}

__forceinline__ __device__ void volatileNodeWriteR(uint32_t* nodeAddress, uint32_t data) {
  volatileWrite(nodeAddress + LANEID_REVERSED(lane_id()), data);
}
__forceinline__ __device__ void volatileNodeWrite(uint32_t* nodeAddress, uint32_t data) {
  volatileWrite(nodeAddress + lane_id(), data);
}

__device__ bool try_acquire_lock(uint32_t* nodeAddress) {
  bool isLocked = true;
  {
    if (lane_id() == 1) {
      isLocked = (atomicOr(nodeAddress + 1, 0x80000000) & 0x80000000) != 0;
    }
    isLocked = __shfl_sync(WARP_MASK, isLocked, 1);
  }
  if (isLocked)
    return true;
  __threadfence();
  return false;
}
__device__ void acquire_lock(uint32_t* nodeAddress) {
  bool isLocked = true;
  while (isLocked) {
    if (lane_id() == 1) {
      isLocked = (atomicOr(nodeAddress + 1, 0x80000000) & 0x80000000) != 0;
    }
    isLocked = __shfl_sync(WARP_MASK, isLocked, 1);
  }
  __threadfence();
}
__device__ void release_lock(uint32_t* nodeAddress) {
  __threadfence();
  if (lane_id() == 1) {
    atomicAnd(nodeAddress + 1, 0x7fffffff);
  }
}

template<typename AllocatorT>
__forceinline__ __device__ void insert_into_node(bool isIntermediate,
                                                 uint32_t next,
                                                 uint32_t src_key,
                                                 uint32_t& src_value,
                                                 AllocatorT*& memAlloc,
                                                 uint32_t src_unit_data = 0) {
  if (isIntermediate)
    src_unit_data = volatileNodeReadR(memAlloc->getAddressPtr(next));

  src_unit_data &= 0x7FFFFFFF;
  src_key = src_key & 0x7FFFFFFF;

  uint32_t insertion_location =
      __ballot_sync(WARP_MASK, (src_key > src_unit_data) && (src_unit_data != 0)) &
      KEY_PIVOT_MASK;
  int dest_lane_pivot = __ffs(insertion_location) - 1;

  if (dest_lane_pivot < 0)
    dest_lane_pivot = 33;

  uint32_t to_move = (1U << (dest_lane_pivot - 3)) - 1U;
  uint32_t lane_mask = (1 << lane_id());

  uint32_t my_new_data =
      __shfl_down_sync(WARP_MASK, src_unit_data, 2, 32);

  to_move = to_move & 0xfffffffc;

  bool shifted = lane_mask & to_move;
  my_new_data = shifted * my_new_data + (1 - shifted) * src_unit_data;

  uint32_t key_loc = (1 << (dest_lane_pivot - 2));
  uint32_t val_loc = (1 << (dest_lane_pivot - 3));

  if (key_loc & lane_mask)
    my_new_data = src_key;
  if (val_loc & lane_mask)
    my_new_data = src_value;

  int valid_location = __ballot_sync(WARP_MASK, src_unit_data != 0);
  valid_location = valid_location >> 2;
  if (isIntermediate && (valid_location & lane_mask & KEY_PIVOT_MASK)) {
    my_new_data = my_new_data | 0x80000000;
  }
  if (lane_id() == 30) {
    my_new_data = (my_new_data | 0x80000000);
  }

  volatileWrite(memAlloc->getAddressPtr(next) + LANEID_REVERSED(lane_id()), my_new_data);
  src_unit_data = my_new_data & 0x7FFFFFFF;
}

template<typename AllocatorT>
__forceinline__ __device__ bool split_node1(uint32_t myParent,
                                            uint32_t src_key,
                                            uint32_t& nodeIdx,
                                            uint32_t& mydata,
                                            AllocatorT*& memAlloc) {
  uint32_t rightData = __shfl_up_sync(0x0000FFFF, mydata, 16);

  if (lane_id() < 18)
    rightData = 0;
  if (lane_id() < 2)
    rightData = mydata;

  uint32_t rightDataMin = __shfl_sync(WARP_MASK, rightData, 31) & 0x7FFFFFFF;

  acquire_lock(memAlloc->getAddressPtr(myParent));

  uint32_t parent_data = volatileNodeReadR(memAlloc->getAddressPtr(myParent));
  uint32_t parent_last_key = __shfl_sync(WARP_MASK, parent_data, 3, 32);

  if (parent_last_key)
  {
    release_lock(memAlloc->getAddressPtr(nodeIdx));
    release_lock(memAlloc->getAddressPtr(myParent));

    return true;
  }

  uint32_t isItMyParent =
      __ballot_sync(WARP_MASK, (parent_data & 0x7FFFFFFF) == nodeIdx) & PIVOT_KEY_MASK;
  if (isItMyParent == 0)
  {
    release_lock(memAlloc->getAddressPtr(nodeIdx));
    release_lock(memAlloc->getAddressPtr(myParent));

    return true;
  }

  uint32_t rightIdx;
  if (!lane_id()) {
    rightIdx = memAlloc->allocate();
  }

  rightIdx = __shfl_sync(WARP_MASK, rightIdx, 0);
  acquire_lock(memAlloc->getAddressPtr(rightIdx));

  insert_into_node(true, myParent, rightDataMin, rightIdx, memAlloc, parent_data);

  release_lock(memAlloc->getAddressPtr(myParent));

  if (lane_id() == 30)
    rightData |= 0x80000000;
  volatileWrite(memAlloc->getAddressPtr(rightIdx) + LANEID_REVERSED(lane_id()),
                rightData);

  if (lane_id() < 16)
    mydata = 0;

  if (lane_id() == 1)
    mydata = rightDataMin;
  if (lane_id() == 0)
    mydata = rightIdx;

  if (lane_id() == 30)
    mydata |= 0x80000000;

  volatileWrite(memAlloc->getAddressPtr(nodeIdx) + LANEID_REVERSED(lane_id()), mydata);

  if (src_key >= rightDataMin)
  {
    release_lock(memAlloc->getAddressPtr(nodeIdx));

    mydata = rightData;
    nodeIdx = rightIdx;
  } else {
    release_lock(memAlloc->getAddressPtr(rightIdx));
  }
  mydata &= 0x7FFFFFFF;

  return false;
}

template<typename AllocatorT>
__forceinline__ __device__ void split_root_node(uint32_t src_key,
                                                uint32_t& nodeIdx,
                                                uint32_t& mydata,
                                                AllocatorT*& memAlloc) {
  uint32_t leftIdx, rightIdx;

  if (!lane_id()) {
    leftIdx = memAlloc->allocate();
    rightIdx = memAlloc->allocate();
  }

  leftIdx = __shfl_sync(WARP_MASK, leftIdx, 0);
  rightIdx = __shfl_sync(WARP_MASK, rightIdx, 0);

  acquire_lock(memAlloc->getAddressPtr(leftIdx));
  acquire_lock(memAlloc->getAddressPtr(rightIdx));

  uint32_t rootData;
  rootData = __shfl_sync(0xFFFFFFFF, mydata, 15) | 0x80000000;
  if (LANEID_REVERSED(lane_id()) == 0)
    rootData = mydata | 0x80000000;
  else if (LANEID_REVERSED(lane_id()) == 1)
    rootData = leftIdx | 0x80000000;
  else if (LANEID_REVERSED(lane_id()) == 3)
    rootData = rightIdx;
  else if (LANEID_REVERSED(lane_id()) == 2)
    rootData = rootData;
  else
    rootData = 0;

  volatileWrite(memAlloc->getAddressPtr(nodeIdx) + LANEID_REVERSED(lane_id()), rootData);
  release_lock(memAlloc->getAddressPtr(nodeIdx));

  uint32_t rightDataMin = __shfl_sync(0xFFFFFFFF, rootData, 29) & 0x7fffffff;

  uint32_t leftData = mydata;
  if (lane_id() < 16)
    leftData = 0;
  if (lane_id() == 1)
    leftData = rightDataMin;
  if (lane_id() == 0)
    leftData = rightIdx;

  if (lane_id() < 16) {
    if (lane_id() == 14)
      mydata |= 0x80000000;
    volatileWrite(memAlloc->getAddressPtr(rightIdx) + +15 - lane_id(), mydata);
  }

  if (lane_id() == 30)
    leftData |= 0x80000000;

  volatileWrite(memAlloc->getAddressPtr(leftIdx) + LANEID_REVERSED(lane_id()), leftData);

  if (LANEID_REVERSED(lane_id()) == 0)
    rootData = (rootData & 0x7fffffff);
  if (LANEID_REVERSED(lane_id()) == 2)
    rootData = (rootData & 0x7fffffff);
  bool goRight = true;
  if (LANEID_REVERSED(lane_id()) == 2 && src_key < rootData)
    goRight = false;

  goRight = __shfl_sync(WARP_MASK, goRight, 29, 32);

  if (goRight) {
    release_lock(memAlloc->getAddressPtr(leftIdx));

    mydata = __shfl_up_sync(0x0000FFFF, mydata, 16);
    if (lane_id() < 16)
      mydata = 0;
    nodeIdx = rightIdx;
  } else {
    release_lock(memAlloc->getAddressPtr(rightIdx));
    mydata = leftData;
    nodeIdx = leftIdx;
  }
  mydata &= 0x7FFFFFFF;
}

template<typename KeyT, typename ValueT, typename AllocatorT>
__device__ void insertion_unit(bool& to_be_inserted,
                               KeyT& myKey,
                               ValueT& myValue,
                               uint32_t*& d_root,
                               AllocatorT* memAlloc) {
  uint32_t work_queue;
  uint32_t last_work_queue = 0;
  uint32_t rootAddress = *d_root;
  uint32_t parent = rootAddress;
  uint32_t next = rootAddress;

  while ((work_queue = __ballot_sync(WARP_MASK, to_be_inserted))) {
    char FullLeafLinkRoot = 0;
    uint32_t src_key = __shfl_sync(WARP_MASK, myKey, __ffs(work_queue) - 1, 32);

    if (last_work_queue != work_queue) {
      next = rootAddress;
      parent = rootAddress;
    }

    uint32_t src_unit_data = volatileNodeReadR(memAlloc->getAddressPtr(next));

    uint32_t link_min =
        __shfl_sync(WARP_MASK, src_unit_data, 1, 32) & 0x7FFFFFFF;

    while (link_min && src_key >= link_min)
    {
      next = __shfl_sync(WARP_MASK, src_unit_data, 0, 32) & 0x7FFFFFFF;
      src_unit_data = volatileNodeReadR(memAlloc->getAddressPtr(next));
      ;
      link_min =
          __shfl_sync(WARP_MASK, src_unit_data, 1, 32) & 0x7FFFFFFF;
      FullLeafLinkRoot |= 0x2;
    }

    uint32_t first_key = __shfl_sync(WARP_MASK, src_unit_data, 31, 32);
    FullLeafLinkRoot =
        ((first_key & 0x80000000) == 0) ? FullLeafLinkRoot | 0x4 : FullLeafLinkRoot & 0x3;

    if (FullLeafLinkRoot & 0x4) {
      if (try_acquire_lock(memAlloc->getAddressPtr(next))) {
        next = parent;
        continue;
      }
      src_unit_data = volatileNodeReadR(memAlloc->getAddressPtr(next));

      first_key = __shfl_sync(WARP_MASK, src_unit_data, 31, 32);
      FullLeafLinkRoot = ((first_key & 0x80000000) == 0) ? FullLeafLinkRoot | 0x4
                                                         : FullLeafLinkRoot & 0x3;

      link_min =
          __shfl_sync(WARP_MASK, src_unit_data, 1, 32) & 0x7FFFFFFF;

      if ((parent == rootAddress) && link_min && src_key >= link_min) {
        release_lock(memAlloc->getAddressPtr(next));

        next = rootAddress;
        parent = rootAddress;
        continue;
      }

      if (!(FullLeafLinkRoot & 0x4))
        release_lock(memAlloc->getAddressPtr(next));

      while (link_min && src_key >= link_min) {
        if (FullLeafLinkRoot & 0x4)
          release_lock(memAlloc->getAddressPtr(next));
        next = __shfl_sync(WARP_MASK, src_unit_data, 0, 32) & 0x7FFFFFFF;
        if (FullLeafLinkRoot & 0x4)
          acquire_lock(memAlloc->getAddressPtr(next));

        src_unit_data = volatileNodeReadR(memAlloc->getAddressPtr(next));

        first_key = __shfl_sync(WARP_MASK, src_unit_data, 31, 32);
        FullLeafLinkRoot = ((first_key & 0x80000000) == 0) ? FullLeafLinkRoot | 0x4
                                                           : FullLeafLinkRoot & 0x3;

        if (!(FullLeafLinkRoot & 0x4))
          release_lock(memAlloc->getAddressPtr(next));

        link_min = __shfl_sync(WARP_MASK, src_unit_data, 1, 32) &
                   0x7FFFFFFF;
        FullLeafLinkRoot |= 0x2;
      }
    }

    FullLeafLinkRoot = __shfl_sync(WARP_MASK, src_unit_data, 3, 32)
                           ? FullLeafLinkRoot | 0x8
                           : FullLeafLinkRoot & 0x7;
    if ((FullLeafLinkRoot & 0x2) && (FullLeafLinkRoot & 0x8)) {
      if (FullLeafLinkRoot & 0x4) {
        release_lock(memAlloc->getAddressPtr(next));
        next = rootAddress;
        parent = rootAddress;
        continue;
      } else {
        FullLeafLinkRoot &= 0x7;
      }
    }

    if ((FullLeafLinkRoot & 0x8) &&
        !(FullLeafLinkRoot & 0x4))
    {
      if (try_acquire_lock(memAlloc->getAddressPtr(next))) {
        next = parent;
        continue;
      } else {
        src_unit_data = volatileNodeReadR(memAlloc->getAddressPtr(next));

        FullLeafLinkRoot = __shfl_sync(WARP_MASK, src_unit_data, 3, 32)
                               ? FullLeafLinkRoot | 0x8
                               : FullLeafLinkRoot & 0x7;

        if (FullLeafLinkRoot & 0x8)
        {
          link_min =
              __shfl_sync(WARP_MASK, src_unit_data, 1, 32) & 0x7FFFFFFF;
          if (link_min && src_key >= link_min)
          {
            release_lock(memAlloc->getAddressPtr(next));
            next = rootAddress;
            parent = rootAddress;
            continue;
          }
        } else {
          release_lock(memAlloc->getAddressPtr(next));

          link_min =
              __shfl_sync(WARP_MASK, src_unit_data, 1, 32) & 0x7FFFFFFF;
          while (link_min && src_key >= link_min)
          {
            next = __shfl_sync(WARP_MASK, src_unit_data, 0, 32) &
                   0x7FFFFFFF;
            src_unit_data = volatileNodeReadR(memAlloc->getAddressPtr(next));
            link_min = __shfl_sync(WARP_MASK, src_unit_data, 1, 32) &
                       0x7FFFFFFF;
            FullLeafLinkRoot |= 0x2;
          }
        }
      }
    }

    if ((FullLeafLinkRoot & 0x8) && (next != rootAddress) && (parent == next)) {
      release_lock(memAlloc->getAddressPtr(next));
      next = rootAddress;
      parent = rootAddress;
      continue;
    }

    if ((FullLeafLinkRoot & 0x8) && (next != rootAddress)) {
      if (split_node1(parent, src_key, next, src_unit_data, memAlloc)) {
        next = rootAddress;
        parent = rootAddress;
        continue;
      }
      if (!(FullLeafLinkRoot & 0x4))
        release_lock(memAlloc->getAddressPtr(next));
    } else if ((FullLeafLinkRoot & 0x8)) {
      split_root_node(src_key, next, src_unit_data, memAlloc);
      FullLeafLinkRoot |= 0x1;
      if (!((FullLeafLinkRoot & 0x4)))
        release_lock(memAlloc->getAddressPtr(next));
    }
    parent = (FullLeafLinkRoot & 0x1) ? rootAddress : next;
    if (FullLeafLinkRoot & 0x4) {
      uint32_t src_lane1 = __ffs(work_queue) - 1;
      uint32_t src_value = __shfl_sync(WARP_MASK, myValue, src_lane1, 32);
      bool key_exist =
          __ballot_sync(WARP_MASK, src_key == src_unit_data) & KEY_PIVOT_MASK;
      if (!key_exist)
        insert_into_node(false, next, src_key, src_value, memAlloc, src_unit_data);
      release_lock(memAlloc->getAddressPtr(next));
      if (src_lane1 == lane_id())
        to_be_inserted = false;
    } else {
      src_unit_data = src_unit_data ? src_unit_data : 0xFFFFFFFF;
      uint32_t src_unit_key = src_unit_data & 0x7FFFFFFF;

      next = __ballot_sync(WARP_MASK, src_key >= (src_unit_key)) & KEY_PIVOT_MASK;
      next = __ffs(next);
      if (next == 0)
        next = __shfl_sync(WARP_MASK, src_unit_key, 30, 32);
      else
        next = __shfl_sync(WARP_MASK, src_unit_key, next - 2, 32);
    }
    last_work_queue = work_queue;
  }
}

//////////////////
////// Search  ///
//////////////////
#define SEARCH_NOT_FOUND 0

template<typename AllocatorT>
__device__ void search_unit(bool& to_be_searched,
                            uint32_t& laneId,
                            uint32_t& myKey,
                            uint32_t& myResult,
                            uint32_t*& d_root,
                            AllocatorT* memAlloc) {
  uint32_t rootAddress = *d_root;

  uint32_t landId_reversed = 31 - laneId;
  uint32_t work_queue = 0;
  uint32_t last_work_queue = 0;
  uint32_t next = rootAddress;

  while ((work_queue = __ballot_sync(WARP_MASK, to_be_searched))) {
    uint32_t src_lane = __ffs(work_queue) - 1;
    uint32_t src_key = __shfl_sync(WARP_MASK, myKey, src_lane, 32);

    bool found = false;
    next = (last_work_queue != work_queue)
               ? rootAddress
               : next;

    uint32_t src_unit_data = *(memAlloc->getAddressPtr(next) + landId_reversed);

    bool isLeaf = ((src_unit_data & 0x80000000) == 0);
    isLeaf = __shfl_sync(
        WARP_MASK,
        isLeaf,
        31,
        32);

    src_unit_data = src_unit_data ? src_unit_data
                                  : 0xFFFFFFFF;

    uint32_t src_unit_key = src_unit_data & 0x7FFFFFFF;

    uint32_t isFoundPivot_bmp =
        __ballot_sync(WARP_MASK, src_key >= src_unit_key) & KEY_PIVOT_MASK;
    int dest_lane_pivot = __ffs(isFoundPivot_bmp) - 1;

    if (dest_lane_pivot < 0) {
      if (laneId == src_lane) {
        myResult = SEARCH_NOT_FOUND;
        to_be_searched = false;
      }
    } else {
      next = __shfl_sync(WARP_MASK, src_unit_data, dest_lane_pivot - 1, 32);
      found = (isLeaf && src_unit_data == src_key);
      found = __shfl_sync(WARP_MASK, found, dest_lane_pivot, 32);

      if (found && (laneId == src_lane)) {
        myResult = next;
        to_be_searched = false;
      }

      if (isLeaf && !found && (laneId == src_lane)) {
        myResult = SEARCH_NOT_FOUND;
        to_be_searched = false;
      }
    }
    last_work_queue = work_queue;
  }
}

#define KEY_PIVOT_MASK_R 0x15555555

template<typename KeyT, typename AllocatorT>
__device__ void delete_unit_bulk(uint32_t& laneId,
                                 KeyT& myKey,
                                 uint32_t*& d_root,
                                 AllocatorT* memAlloc) {
  int dest_lane_pivot;
  uint32_t rootAddress = *d_root;

#pragma unroll
  for (int src_lane = 0; src_lane < WARP_WIDTH; src_lane++) {
    KeyT src_key = __shfl_sync(WARP_MASK, myKey, src_lane, 32);
    KeyT next = rootAddress;
    bool isIntermediate = true;
    do {
      KeyT src_unit_data = *(memAlloc->getAddressPtr(next) + laneId);
      isIntermediate = !((src_unit_data & 0x80000000) == 0);
      isIntermediate = __shfl_sync(WARP_MASK, isIntermediate, 0, 32);
      if (!isIntermediate) {
        acquire_lock(memAlloc->getAddressPtr(next));
        src_unit_data = volatileNodeRead(memAlloc->getAddressPtr(next));
      }
      KeyT src_unit_key = src_unit_data & 0x7FFFFFFF;
      bool hit = (src_key >= src_unit_key) && src_unit_key;
      bool key_exist =
          __ballot_sync(WARP_MASK, src_key == src_unit_key) & KEY_PIVOT_MASK_R;
      uint32_t isFoundPivot_bmp = __ballot_sync(WARP_MASK, hit);
      dest_lane_pivot = __ffs(~isFoundPivot_bmp & KEY_PIVOT_MASK_R);
      if (isIntermediate) {
        dest_lane_pivot = dest_lane_pivot ? dest_lane_pivot - 2 : 29;
        next = __shfl_sync(WARP_MASK, src_unit_data, dest_lane_pivot, 32);
      } else {
        if (key_exist) {
          uint32_t newNodeData = __shfl_down_sync(WARP_MASK, src_unit_key, 2, 32);
          isFoundPivot_bmp &= KEY_PIVOT_MASK_R;
          isFoundPivot_bmp |= (isFoundPivot_bmp << 1);
          isFoundPivot_bmp >>= 2;
          bool to_move = ((1 << laneId) & ~isFoundPivot_bmp) && (laneId < 30);
          KeyT finalData = (to_move * newNodeData + (!to_move) * src_unit_key);
          finalData = (laneId >= 28 && laneId < 30) ? 0 : finalData;
          finalData = (laneId == 1) ? finalData | 0x80000000 : finalData;
          volatileNodeWrite(memAlloc->getAddressPtr(next), finalData);
        }
        release_lock(memAlloc->getAddressPtr(next));
      }
    } while (isIntermediate);
  }
}

template<typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
__device__ void range_unit(uint32_t& laneId,
                           bool& to_search,
                           KeyT& lower_bound,
                           KeyT& upper_bound,
                           ValueT*& range_results,
                           uint32_t*& d_root,
                           SizeT& range_length,
                           AllocatorT* memAlloc) {
  int dest_lane_pivot;
  uint32_t rootAddress = *d_root;

  while (auto work_queue = __ballot_sync(WARP_MASK, to_search)) {
    auto src_lane = __ffs(work_queue) - 1;
    KeyT src_key_lower = __shfl_sync(WARP_MASK, lower_bound, src_lane, 32);
    KeyT src_key_upper = __shfl_sync(WARP_MASK, upper_bound, src_lane, 32);
    KeyT next = rootAddress;
    bool is_intermediate = true;
    if (laneId == src_lane)
      to_search = false;
    do {
      uint32_t src_unit_data = *(memAlloc->getAddressPtr(next) + laneId);
      is_intermediate = !((src_unit_data & 0x80000000) == 0);
      is_intermediate = __shfl_sync(WARP_MASK, is_intermediate, 0, 32);

      uint32_t src_unit_key = src_unit_data & 0x7FFFFFFF;
      bool hit = (src_key_lower >= src_unit_key) && src_unit_key;
      uint32_t isFoundPivot_bmp = __ballot_sync(WARP_MASK, hit);
      dest_lane_pivot = __ffs(~isFoundPivot_bmp & KEY_PIVOT_MASK_R);
      if (is_intermediate) {
        dest_lane_pivot = dest_lane_pivot ? dest_lane_pivot - 2 : 29;
        next = __shfl_sync(WARP_MASK, src_unit_data, dest_lane_pivot, 32);
      } else {
        uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        tid /= 32;
        tid *= 32;
        tid += src_lane;
        uint32_t offset = tid * range_length * 2;
        while (true) {
          hit = ((src_key_lower <= src_unit_key && src_key_upper >= src_unit_key) &&
                 src_unit_key);
          isFoundPivot_bmp = __ballot_sync(WARP_MASK, hit);
          isFoundPivot_bmp &= KEY_PIVOT_MASK_R;
          dest_lane_pivot = __ffs(isFoundPivot_bmp);

          dest_lane_pivot--;
          isFoundPivot_bmp >>= dest_lane_pivot;
          uint32_t link_min =
              __shfl_sync(WARP_MASK, src_unit_key, 30, 32) & 0x7FFFFFFF;
          src_unit_key = __shfl_down_sync(WARP_MASK, src_unit_key, dest_lane_pivot, 32);
          uint32_t counter = __popc(isFoundPivot_bmp) * 2;
          if (laneId < counter)
            range_results[offset + laneId] = src_unit_key - 2;
          if (!link_min || src_key_upper < link_min)
            break;
          offset += counter;
          next = __shfl_sync(WARP_MASK, src_unit_key, 31, 32) & 0x7FFFFFFF;
          src_unit_key = *(memAlloc->getAddressPtr(next) + laneId);
        }
      }
    } while (is_intermediate);
  }
}
}  // namespace warps
}  // namespace GpuBTree