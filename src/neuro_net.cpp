#include "neuro_net.h"


/*ОПИСАНИЕ ОДНОГО НЕЙРОНА*/
    Neuron::Neuron(char function, double value):
        value(value),
        active_value(1),
        error(0),
        function(function)
    {}
    void Neuron::set_function(std::string const func){
        if(func == "sigma")
            function = 0;
        if(func == "tangens")
            function = 1;
        if(func == "ReLu")
            function = 2;
    }
    void Neuron::sigma(bool is_derive){
        if(is_derive)
            error *= active_value * (1 - active_value);
        else
            active_value = 1 / (1 + exp(-value));
    }
    void Neuron::tangens(bool is_derive){
        if(is_derive)
            error *= (active_value + 1) * (1 - active_value);
        else
            active_value = 2 / (1 + exp(-2 * value)) - 1;
    }
    void Neuron::ReLu(bool is_derive){
        if(is_derive)
            error *= value > 0 ? 1:0;
        else
            active_value = std::max(0.0, value);
    }
    void Neuron::active(bool is_derive){
        if(function == 0)
            sigma(is_derive);
        if(function == 1)
            tangens(is_derive);
        if(function == 2)
            ReLu(is_derive);
    }





/*ОПИСАНИЕ СЕТИ*/
    Net::Net(std::vector<size_t>& NNeurons_in, double etta, double moment):
        NLayers(NNeurons_in.size()),
        etta(etta),
        moment(moment),
        NNeurons(NNeurons_in)
    {
        /*инициализация сети слоями + нейронами*/

        /*плюс нейрон для свободного члена*/
        for(size_t layer = 0; layer < NLayers - 1; ++layer)
            NNeurons[layer]++;

        /*заполнение связей и нейронов*/
        for(size_t layer = 0; layer < NLayers - 2; ++layer){
            std::vector<Neuron> layer_fill(NNeurons[layer] - 1);
            layer_fill.push_back(1);
            Neurons.push_back(layer_fill);

            std::vector<std::vector<double>> link;
            std::vector<std::vector<double>> delta_link;
            for(size_t line = 0; line < NNeurons[layer]; ++line){
                std::vector<double> line_fill;
                for(size_t elem = 0; elem < NNeurons[layer + 1] - 1; ++elem){
                    double fix = 1.0 * (rand() % 10000) / 10000 - 0.5;
                    line_fill.push_back(fix);
                }
                std::vector<double> delta_fill(NNeurons[layer + 1] - 1, 0);
                link.push_back(line_fill);
                delta_link.push_back(delta_fill);
            }
            Links.push_back(link);
            delta_Links.push_back(delta_link);
        }

        /*особенности со свободным членом*/
        std::vector<Neuron> layer_fill(NNeurons[NLayers - 2] - 1);
        layer_fill.push_back(1);
        Neurons.push_back(layer_fill);
        std::vector<Neuron> buf(NNeurons[NLayers - 1]);
        Neurons.push_back(buf);

        std::vector<std::vector<double>> link;
        std::vector<std::vector<double>> delta_link;
        for(size_t line = 0; line < NNeurons[NLayers - 2]; ++line){
            std::vector<double> line_fill;
            for(size_t elem = 0; elem < NNeurons[NLayers - 1]; ++elem){
                double fix = 1.0 * (rand() % 10000) / 10000 - 0.5;
                line_fill.push_back(fix);
            }
            std::vector<double> delta_fill(NNeurons[NLayers - 1], 0);
            link.push_back(line_fill);
            delta_link.push_back(delta_fill);
        }
        Links.push_back(link);
        delta_Links.push_back(delta_link);
        find_max_layer();
    }

    Net::Net(char const* infile_name, double etta, double moment):
        etta(etta),
        moment(moment)
    {
        std::ifstream in(infile_name, std::ios::in);
        if(!in.is_open()){
            std::cout << "Can't open infile" << std::endl;
        } else {
            in >> NLayers;
            for(size_t matrix = 0; matrix < NLayers; ++matrix){
                size_t n_cur, n_next;
                in >> n_cur >> n_next;
                NNeurons.push_back(n_cur);
                std::vector<std::vector<double>> matrix_fill;
                std::vector<std::vector<double>> deltamatrix_fill;
                for(size_t line = 0; line < n_cur; ++line){
                    std::vector<double> line_fill;
                    for(size_t elem = 0; elem < n_next; ++elem){
                        double buf_elem;
                        in >> buf_elem;
                        line_fill.push_back(buf_elem);
                    }
                    deltamatrix_fill.push_back(std::vector<double>(n_next, 0));
                    matrix_fill.push_back(line_fill);
                }
                delta_Links.push_back(deltamatrix_fill);
                Links.push_back(matrix_fill);
            }
            NLayers++;
            NNeurons.push_back(Links[NLayers - 2][0].size());
            for(auto nneuro : NNeurons){
                std::vector<Neuron> Neurons_fill(nneuro);
                Neurons.push_back(Neurons_fill);
            }
        }
    }

    void Net::set_learning_rate(double etta){
        this->etta = etta;
    }

    void Net::set_inertion(double moment){
        this->moment = moment;
    }

    bool Net::download_data(std::vector<double>& in){
        /*загружаем входной слой*/

        if(in.size() + 1 != NNeurons[0]){
            std::cout << "Fix pls: Download" << std::endl;
            return false;
        }
        for(size_t neuro = 0; neuro < in.size(); ++neuro){
            Neurons[0][neuro].active_value = in[neuro];
        }
        return true;
    }

    void Net::forward_pass(){
        /*прямой проход по сети*/

        for(size_t layer = 0; layer < NLayers - 1; ++layer){
            for(size_t new_neuro = 0; new_neuro < NNeurons[layer + 1] - 1; ++new_neuro){
                Neurons[layer + 1][new_neuro].value = 0;
                for(size_t old_neuro = 0; old_neuro < NNeurons[layer]; ++old_neuro){
                    Neurons[layer + 1][new_neuro].value +=
                            Neurons[layer][old_neuro].active_value * Links[layer][old_neuro][new_neuro];
                }
                Neurons[layer + 1][new_neuro].active();
            }
        }
        for(size_t new_neuro = 0; new_neuro < NNeurons[NLayers - 1]; ++new_neuro){
            Neurons[NLayers - 1][new_neuro].value = 0;
            for(size_t old_neuro = 0; old_neuro < NNeurons[NLayers - 2]; ++old_neuro){
                Neurons[NLayers - 1][new_neuro].value +=
                        Neurons[NLayers - 2][old_neuro].active_value * Links[NLayers - 2][old_neuro][new_neuro];
            }
            Neurons[NLayers - 1][new_neuro].active();
        }
    }

    void Net::back_propagation(std::vector<double> result){
        /*обучаем выходным вектором*/

        if(result.size() != Neurons[NLayers - 1].size()){
            std::cout << "Error::back_propagation" << std::endl;
            return;
        }

        /*ищем дельты*/
        for(size_t neuron = 0; neuron < NNeurons[NLayers - 1]; ++neuron){
            Neurons[NLayers - 1][neuron].error = result[neuron] - Neurons[NLayers - 1][neuron].active_value;
            Neurons[NLayers - 1][neuron].active(true);
        }
        for(size_t neuro = 0; neuro < NNeurons[NLayers - 2] - 1; ++neuro){
            Neurons[NLayers - 2][neuro].error = 0;
            for(size_t errneuro = 0; errneuro < NNeurons[NLayers - 1]; ++errneuro){
                Neurons[NLayers - 2][neuro].error +=
                        Neurons[NLayers - 1][errneuro].error * Links[NLayers - 2][neuro][errneuro];
            }
            Neurons[NLayers - 2][neuro].active(true);
        }
        for(size_t layer = NLayers - 3; layer > 0; --layer){
            for(size_t neuro = 0; neuro < NNeurons[layer] - 1; ++neuro){
                Neurons[layer][neuro].error = 0;
                for(size_t errneuro = 0; errneuro < NNeurons[layer + 1] - 1; ++errneuro){
                    Neurons[layer][neuro].error +=
                            Neurons[layer + 1][errneuro].error * Links[layer][neuro][errneuro];
                }
                Neurons[layer][neuro].active(true);
            }
        }
        /*нашли все дельты*/

        /*меняем веса*/
        for(size_t layer = 0; layer < NLayers - 2; ++layer){
            for(size_t prev_neuro = 0; prev_neuro < NNeurons[layer]; ++prev_neuro){
                for(size_t neuro = 0; neuro < NNeurons[layer + 1] - 1; ++neuro){
                    delta_Links[layer][prev_neuro][neuro] *= moment;
                    delta_Links[layer][prev_neuro][neuro] +=
                            Neurons[layer][prev_neuro].active_value * Neurons[layer + 1][neuro].error * etta;
                    Links[layer][prev_neuro][neuro] += delta_Links[layer][prev_neuro][neuro];
                }
            }
        }
        for(size_t prev_neuro = 0; prev_neuro < NNeurons[NLayers - 2]; ++prev_neuro){
            for(size_t neuro = 0; neuro < NNeurons[NLayers - 1]; ++neuro){
                delta_Links[NLayers - 2][prev_neuro][neuro] *= moment;
                delta_Links[NLayers - 2][prev_neuro][neuro] +=
                        Neurons[NLayers - 2][prev_neuro].active_value * Neurons[NLayers - 1][neuro].error * etta;
                Links[NLayers - 2][prev_neuro][neuro] += delta_Links[NLayers - 2][prev_neuro][neuro];
            }
        }
        /*изменили все веса*/
    }

    void Net::dump_Links(char const* outfile_name){
        std::ofstream out(outfile_name, std::ios::out);
        if(!out.is_open()){
            std::cerr << "Can't open file for dump" << std::endl;
            return;
        }
        out << NLayers - 1 << std::endl;
        for(auto matrix : Links){
            out << matrix.size() << ' ' << matrix[0].size() << std::endl;
            for(auto line : matrix){
                for(auto elem : line){
                    out << elem << ' ';
                }
                out << std::endl;
            }
        }
        out.close();
    }

    std::vector<double> Net::get_result(){
        /*выгружаем выходной слой*/

        std::vector<double> result;
        for(auto Neuro : Neurons[NLayers - 1]){
            result.push_back(Neuro.active_value);
        }
        return result;
    }



    void Net::print_Neurons() const{
        /*вывод нейронов сети*/

        std::cout << "Data: Neurons X NLayers(val, er):" << std::endl;
        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        for(size_t neuro = 0; neuro < Max_layer_size; ++neuro){
            for(size_t layer = 0; layer < NLayers; ++layer){
                if(NNeurons[layer] > neuro){
                    std::cout << std::setw(4) << std::right <<'('
                         << std::setw(6) << std::left << Neurons[layer][neuro].active_value
                         << ','
                         << std::setw(6) << std::left << Neurons[layer][neuro].error
                         << ')';
                } else {
                    std::cout << std::setw(18) << ' ';
                }
            }
            std::cout << std::endl;
        }
        std::cout << "\n\n";
        std::cout.unsetf(std::ios::fixed);
    }

    void Net::print_Links() const{
        /*вывод синапсов сети*/

        std::cout.precision(2);
        for(size_t layer = 0; layer < NLayers - 2; ++layer){
            std::cout << "Links between layer_" << layer + 1 << " and layer_" << layer + 2 << std::endl;
            for(size_t lines = 0; lines < NNeurons[layer]; ++lines){
                for(size_t columns = 0; columns < NNeurons[layer + 1] - 1; ++columns){
                    std::cout << Links[layer][lines][columns] << '\t';
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << "Links between layer_" << NLayers - 1 << " and layer_" << NLayers << std::endl;
        for(size_t lines = 0; lines < NNeurons[NLayers - 2]; ++lines){
            for(size_t columns = 0; columns < NNeurons[NLayers - 1]; ++columns){
                std::cout << Links[NLayers - 2][lines][columns] << '\t';
            }
            std::cout << std::endl;
        }
        std::cout << "\n\n";
    }

    void Net::find_max_layer(){ //вычисляет размер наибольшего слоя
        Max_layer_size = 0;
        for(auto layer_size : NNeurons){
            if(layer_size > Max_layer_size)
                Max_layer_size = layer_size;
        }
    }
