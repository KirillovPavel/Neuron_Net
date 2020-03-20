#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <vector>

struct Object{
    std::vector<double> in;
    std::vector<double> out;
public:
    Object();
    Object(std::vector<double> const* in);
    Object(std::vector<double> const& in);
    Object(std::vector<double> const* in, std::vector<double> const* out);
    Object(std::vector<double> const& in, std::vector<double> const& out);
    void replace_in(std::vector<double> const in_vec);
    void replace_out(std::vector<double> const out_vec);
};

struct Data{
    std::vector<Object> data;
public:
    Data();
    Data(Object line);
    Data(Data const* another);
    Object operator[](size_t i);
    void push_back(Object line);
    void push_back(std::vector<double> const* in, std::vector<double> const* out);
    void erase(size_t i);
};

#endif // DATAFRAME_H
