#include "data_gen.h"
#include <cstdlib>
#include <ctime>


static bool initialized = false;


double random_fixed_4digit_double() {
    int raw = rand() % 10000000; 
    return static_cast<double>(raw) / 10000.0;
}

Val generate_random_data() {
    if (!initialized) {
        srand(static_cast<unsigned int>(time(nullptr)));
        initialized = true;
    }

    Val json_val;
    json_val.type = DICT;
    json_val.put("Temp", random_fixed_4digit_double());
    json_val.put("Press", random_fixed_4digit_double());

    return json_val;
}
