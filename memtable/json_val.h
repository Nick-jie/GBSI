#ifndef JSON_VAL_H
#define JSON_VAL_H

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>

#define INT 0
#define DOU 1
#define STR 2
#define BOOL 3
#define LIST 4
#define DICT 5

class Val {
public:
    int INT_VAL;
    double DOU_VAL;
    std::string STR_VAL;
    bool BOOL_VAL;
    std::vector<Val> List;
    std::map<Val, Val> dict;
    int type;

    Val();
    ~Val();
    Val(const int& x);
    Val(const double& x);
    Val(const std::string& x);
    Val(const char* x);
    Val(const bool& x);
    Val(std::initializer_list<Val> l);

    void del();
    void add(Val x);
    void put(Val key, Val val);
    Val& operator[](Val i);

    std::string str();
};

std::ostream& operator<<(std::ostream& out, const Val& v);
bool operator<(const Val& a, const Val& b);

Val parser(const std::string& s);

#endif
