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

void bin(int a, std::vector<double>& res){
	for(int i = 0; i < 8; ++i){
		res.push_back(a % 2);
		a = a / 2;
	}
}

using namespace std;

int main()
{
    vector<size_t> nneurons = {1, 6, 5, 7, 8};
    Net net(nneurons, 0.6, 0.6);
    Data data;
    for(int i = 0; i < 256; ++i){
	    vector<double> buf;
	    bin(i, buf);
	    vector<double> in(i);
	    Object obj(in, buf);
	    data.push_back(obj);
    }
    learn(data, net, 5000);
    net.dump_links("y_x");

    //Net net("y_x");
    Object obj({3});
    result(obj, net);
    for(auto i : obj.out)
        cout << i << ' ';
    return 0;
}
