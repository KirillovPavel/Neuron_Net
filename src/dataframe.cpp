#include "dataframe.h"
#include <iostream>

/*ОПИСАНИЕ ОБЪЕКТА*/
    Object::Object(){}
    Object::Object(std::vector<double> const* in):
        in(*in)
    {}
    Object::Object(std::vector<double> const& in):
        in(in)
    {}
    Object::Object(std::vector<double> const* in, std::vector<double> const* out):
        in(*in),
        out(*out)
    {}
    Object::Object(std::vector<double> const& in, std::vector<double> const& out):
        in(in),
        out(out)
    {}
    void Object::replace_in(std::vector<double> const in_vec){
        in = in_vec;
    }
    void Object::replace_out(std::vector<double> const out_vec){
        out = out_vec;
    }

/*ОПИСАНИЕ DATAFRAME'a*/
    Data::Data(){}
    Data::Data(Object line){
        data.push_back(line);
    }
    Data::Data(Data const* another){
        data = another->data;
    }
    Object Data::operator[](size_t i){
        return data[i];
    }
    void Data::push_back(Object line){
        data.push_back(line);
    }
    void Data::push_back(std::vector<double> const* in, std::vector<double> const* out){
        data.push_back(Object(in, out));
    }
    void Data::erase(size_t i){
        if(i >= data.size()){
            std::cerr << "Error:: Data.erase()" << std::endl;
            return;
        }
        data.erase(data.begin() + i);
    }
