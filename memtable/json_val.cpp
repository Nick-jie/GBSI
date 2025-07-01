#include "json_val.h"
#include <cctype>

Val::Val() {}
Val::~Val() { del(); }

Val::Val(const int& x) { del(); INT_VAL = x; type = INT; }
Val::Val(const double& x) { del(); DOU_VAL = x; type = DOU; }
Val::Val(const std::string& x) { del(); STR_VAL = x; type = STR; }
Val::Val(const char* x) { del(); STR_VAL = std::string(x); type = STR; }
Val::Val(const bool& x) { del(); BOOL_VAL = x; type = BOOL; }
Val::Val(std::initializer_list<Val> l) { del(); type = LIST; for (Val x : l) List.push_back(x); }

void Val::del() {
    if (type == STR) STR_VAL.clear();
    if (type == LIST) List.clear();
    if (type == DICT) dict.clear();
}

void Val::add(Val x) { if (type == LIST) List.push_back(x); }
void Val::put(Val key, Val val) { if (type == DICT) dict[key] = val; }
Val& Val::operator[](Val i) { return type == LIST ? List[i.INT_VAL] : dict[i]; }

std::string Val::str() {
    std::stringstream ss; ss << (*this);
    return ss.str();
}

std::ostream& operator<<(std::ostream& out, const Val& v) {
    if (v.type == INT) out << v.INT_VAL;
    else if (v.type == DOU) out << v.DOU_VAL;
    else if (v.type == STR) out << "\"" << v.STR_VAL << "\"";
    else if (v.type == BOOL) out << (v.BOOL_VAL ? "true" : "false");
    else if (v.type == LIST) {
        out << "[";
        for (size_t i = 0; i < v.List.size(); i++) {
            if (i) {
                out << ",";
            }
            out << v.List[i];
        }
        out << "]";
    }
    else if (v.type == DICT) {
        out << "{";
        for (auto it = v.dict.begin(); it != v.dict.end(); it++) {
            if (it != v.dict.begin()) out << ",";
            out << it->first << ":" << it->second;
        }
        out << "}";
    }
    return out;
}

bool operator<(const Val& a, const Val& b) {
    if (a.type != b.type) return a.type < b.type;
    if (a.type == INT) return a.INT_VAL < b.INT_VAL;
    if (a.type == DOU) return a.DOU_VAL < b.DOU_VAL;
    if (a.type == STR) return a.STR_VAL < b.STR_VAL;
    if (a.type == LIST) return a.List < b.List;
    if (a.type == DICT) return a.dict < b.dict;
    return true;
}

// ----------------- JSON Parser -------------------

static std::stringstream ss;

Val parser_val();
Val parser_num();
Val parser_str();
Val parser_bool();
Val parser_arr();
Val parser_map();

Val parser_num() {
    std::string s;
    while (isdigit(ss.peek()) || ss.peek() == 'e' || ss.peek() == '.' || ss.peek() == '-' || ss.peek() == '+')
        s.push_back(ss.get());
    return (s.find('.') != std::string::npos || s.find('e') != std::string::npos) ? std::stod(s) : std::stoi(s);
}

Val parser_str() {
    ss.get(); // skip "
    std::string s;
    while (ss.peek() != '"') s.push_back(ss.get());
    ss.get(); // skip "
    return Val(s);
}

Val parser_bool() {
    if (ss.peek() == 'f') { for (int i = 0; i < 5; ++i) ss.get(); return Val(false); }
    else { for (int i = 0; i < 4; ++i) ss.get(); return Val(true); }
}

Val parser_arr() {
    ss.get(); // skip [
    Val vec; vec.type = LIST;
    while (ss.peek() != ']') {
        vec.add(parser_val());
        while (ss.peek() != ']' && (isspace(ss.peek()) || ss.peek() == ',')) ss.get();
    }
    ss.get(); // skip ]
    return vec;
}

Val parser_map() {
    ss.get(); // skip {
    Val dict; dict.type = DICT;
    while (ss.peek() != '}') {
        Val key = parser_val();
        while (isspace(ss.peek()) || ss.peek() == ':') ss.get();
        Val val = parser_val();
        dict.put(key, val);
        while (ss.peek() != '}' && (isspace(ss.peek()) || ss.peek() == ',')) ss.get();
    }
    ss.get(); // skip }
    return dict;
}

Val parser_val() {
    while (ss.peek() != -1) {
        if (isspace(ss.peek())) ss.get();
        else if (ss.peek() == '"') return parser_str();
        else if (ss.peek() == 't' || ss.peek() == 'f') return parser_bool();
        else if (ss.peek() == '[') return parser_arr();
        else if (ss.peek() == '{') return parser_map();
        else return parser_num();
    }
    return Val(0);
}

Val parser(const std::string& s) {
    ss.clear(); ss.str(s);
    return parser_val();
}
