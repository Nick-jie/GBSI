#ifndef DATA_EXPORT_H
#define DATA_EXPORT_H

#include <string>
#include "leveldb/db.h"


void print_parsed_json(const std::string& value_str);


void export_all_forward(leveldb::DB* db);


void export_all_reverse(leveldb::DB* db);


void export_from_key_reverse(leveldb::DB* db, const std::string& start_key);


int count_entries(leveldb::DB* db);

#endif
