#include "read.hpp"

vector<vector<double> > train_images;
vector<int> train_labels;

vector<vector<double> > test_images;
vector<int> test_labels;

const int input_num = 28 * 28;

const int hidden_num = 300;
const int output_num = 10;

const int train_num = 60000;
const int test_num = 10000;
// net hype-parameters
const int batch_size = 10;
const int train_inteval = train_num/batch_size;
double lr = 0.001;
double wei_momentum = 0.1;
enum ac {Sigmoid,Relu,Tanh};
ac activation;
ofstream file;


void read(){
    read_Mnist_Images("data/train-images.idx3-ubyte", train_images);
    read_Mnist_Label("data/train-labels.idx1-ubyte", train_labels);
    read_Mnist_Images("data/test-images.idx3-ubyte", test_images);
    read_Mnist_Label("data/test-labels.idx1-ubyte", test_labels);
}
// net parameters
vector< vector<double > > weight12(hidden_num,vector<double>(input_num,0));
vector<double> bias2(hidden_num,0);

vector< vector<double > > weight23(output_num,vector<double>(hidden_num,0));
vector<double> bias3(output_num,0);

//net data
vector<double> input_layer_data(input_num,0);
vector<double> hidden_layer_data(hidden_num,0);
vector<double> output_layer_data(output_num,0);


vector< vector<double > > diff_weight12(hidden_num,vector<double>(input_num,0));
vector<double> diff_bias2(hidden_num,0);

vector< vector<double > > diff_weight23(output_num,vector<double>(hidden_num,0));
vector<double> diff_bias3(output_num,0);

vector<double> diff_hidden(hidden_num,0);
vector<double> diff_output(output_num,0);

vector<int> shuffle(train_num,0);
void shuffle_data(){
    for(int  i = train_num - 1;i > 0; --i){
        int t = rand()%i;
        swap(shuffle[i],shuffle[t]);
    }
}
void p_net_diff(double p){
    momentum_vec2(diff_weight12,p);
    momentum_vec2(diff_weight23,p);

    momentum_vec1(diff_bias2,p);
    momentum_vec1(diff_bias3,p);
}
void init_net(){
    file.open("log/log_sigmoid300_lambda.txt");
    file << "initial learning rate:" << lr << endl;
    file << "activation function: "  << activation << endl;
    cout << "function list 0: Sigmoid 1:Relu 2:Tanh" << endl;
    cout << "lambda value: " << lambda << endl;
    file << "hidden layer unit number : " << hidden_num << endl << endl << endl;
    activation = Sigmoid;
    init_layer_weight(weight12);
    init_layer_bias(bias2);
    init_layer_weight(weight23);
    init_layer_bias(bias3);
    for(int i = 0; i < train_num; ++i){
        shuffle[i] = i;
    }
    p_net_diff(0);
}


void net_forward(){
    forward(input_layer_data, weight12, bias2, hidden_layer_data);
    if(activation == Sigmoid)
        sigmoid_vec(hidden_layer_data);
    else if(activation == Relu)
        relu(hidden_layer_data);
    else 
        tanh(hidden_layer_data);
    forward(hidden_layer_data, weight23, bias3, output_layer_data);
    minus_max(output_layer_data);

    softmax_vec(output_layer_data);
}

void net_backward(int number){
    for(int i = 0;i < output_num;++i){
        diff_output[i] = output_layer_data[i] - (i == number? 1.0 : 0); 
    }

    for(int i = 0;i < hidden_num; ++i){
        diff_hidden[i] = 0;
        for(int j = 0;j < output_num; ++j){
            if(activation == Sigmoid)
                diff_hidden[i] += diff_output[j] * weight23[j][i] * hidden_layer_data[i] * (1 - hidden_layer_data[i]);
            else if(activation == Relu)
                diff_hidden[i] += diff_output[j] * weight23[j][i] * (hidden_layer_data[i] >= 0? 1:0);
            else
                diff_hidden[i] += diff_output[j] * weight23[j][i] * (1 - hidden_layer_data[i] * hidden_layer_data[i]);
        }
    }


    for(int j = 0;j <output_num; ++j){
        for(int i = 0; i < hidden_num; ++i){
            diff_weight23[j][i] += diff_output[j] * hidden_layer_data[i] / batch_size * wei_momentum;
        }
        diff_bias3[j] += diff_output[j] / batch_size * wei_momentum;
    }

    for(int j = 0;j <hidden_num; ++j){
        for(int i = 0; i < input_num; ++i){
            diff_weight12[j][i] += diff_hidden[j] * input_layer_data[i] / batch_size * wei_momentum;
        }
        diff_bias2[j] += diff_hidden[j] / batch_size * wei_momentum;
    }
}

void update_weight(){

    update_vec2(weight12,diff_weight12,lr);
    update_vec1(bias2,diff_bias2,lr);

    update_vec2(weight23,diff_weight23,lr);
    update_vec1(bias3,diff_bias3,lr);
}
void test(int epoch){
    int hit = 0;
    for(int  i = 0;i < test_num; ++i){
        input_layer_data.swap(test_images[i]);
        net_forward();
        if(find_best(output_layer_data) == test_labels[i])
            ++hit;
        input_layer_data.swap(test_images[i]);
    }
    cout << "****************************************************************" << endl;
    cout << "the epoch times: " << epoch << endl;
    cout << "the learning rate now :  " << lr << endl;
    cout << "the accuracy on test data now :  " << hit*1.0/test_num << endl << endl;

    cout << "****************************************************************" << endl;
    file << "the epoch times: " << epoch << endl;
    file << "the learning rate now :  " << lr << endl;
    file << "the accuracy on test data now :  " << hit*1.0/test_num << endl << endl;
}
void solver(){
    init_net();
    int epoch = 0;
    while(1){
        double right = 0;
        shuffle_data();
        int hit = 0;
        for( int k = 0;k < train_inteval; ++k){
            p_net_diff(1.0 - wei_momentum);
            for(int i = 0;i < batch_size; ++i){
                int t_num = shuffle[k * batch_size + i];
                input_layer_data.swap(train_images[t_num]);
                net_forward();
                if(find_best(output_layer_data) == train_labels[t_num])
                    ++hit;
                net_backward(train_labels[t_num]);
                input_layer_data.swap(train_images[t_num]);
            }
            if( k == 0)
                p_net_diff(1.0 / wei_momentum);
            update_weight();

            if(k%50 == 0){
                cout << "the learning rate now :  " << lr << endl;
                cout << "the accuracy on train data now :  " << hit * 1.0 / (50 * batch_size) << endl;
                file << "the learning rate now :  " << lr << endl;
                file << "the accuracy on train data now :  " << hit * 1.0 / (50 * batch_size) << endl;
                hit = 0;
            }
        }
        ++epoch;
        if(epoch % 20 == 0)
        lr *= 0.1;
        test(epoch);
        if(lr < 1e-7){
            cout << "Optimization done !" << endl;
            file << "Optimization done !" << endl;
            break;
        }
    }
}
int main(){
    read();
    solver();
	return 0;
}

//     Mat src = imread("/home/xiefubao/mnist/images.jpg");  
//     namedWindow("Display");
//     imshow("Display", src);
//     waitKey(0);
//     read();
//     Mat src(28,28,CV_8U,Scalar(0));
//     for(int i = 0;i < width; i++)
//     for(int j = 0;j < height; j++)
//     {
//         src.at<uchar>(i, j) = images[2][i * height + j];
//     }
//     resize(src,src,Size(300,300));