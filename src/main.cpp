#include "neuro_net.h"
#include "dataframe.h"
#include <iostream>
#include <cmath>

void learn(Data const& data, Net& net, int steps){
    for(int i = 0; i < steps; ++i){
        for(auto obj : data.data){
            net.download_data(obj.in);
            net.forward_pass();
            net.back_propagation(obj.out);
        }
    }
}

void result(Object& obj, Net& net){
    net.download_data(obj.in);
    net.forward_pass();
    obj.replace_out(net.get_result());
}



int main()
{
    //XOR:

    //init
    std::vector<size_t> Nneurons = {2, 2, 1};
    Net XOR(Nneurons);
    Data data;
    data.push_back(Object({0, 0}, {1}));
    data.push_back(Object({1, 0}, {0}));
    data.push_back(Object({0, 1}, {0}));
    data.push_back(Object({1, 1}, {1}));

    //learning
    learn(data, XOR, 10000);

    //checking result
    Object check({0, 1}, {0});
    result(check, XOR);
    std::cout << check.out[0] << std::endl;

    XOR.print_Neurons();

    return 0;
}
