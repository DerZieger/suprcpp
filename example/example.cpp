#include "suprcpp.h"
#include "fstream"

int main(){
    torch::Device device=torch::kCPU;//if available torch::kCUDA
    auto a=SUPRCPP("/EXAMPLE_PATH/neutral.npz");
    a.to(device);
    torch::Tensor b=torch::zeros({1,10}).to(device);
    torch::Tensor t=torch::zeros({1,3}).to(device);
    torch::Tensor p=torch::rand({1,a.getNumPose()}).to(device);
    p=p-torch::ones_like(p)*0.5;
    auto o=a.forward(p,b,t);

    std::ofstream MyFile("test.obj");

    for(int i=0;i<o.v_final.size(1);++i){
        MyFile<<"v "<<o.v_final[0][i][0].item<float>()<<" "<<o.v_final[0][i][1].item<float>()<<" "<<o.v_final[0][i][2].item<float>()<<"\n";
    }
    for(int i=0;i<o.f.size(0);++i){
        MyFile<<"f "<<(o.f[i][0].item<long>()+1)<<" "<<(o.f[i][1].item<long>()+1)<<" "<<(o.f[i][2].item<long>()+1)<<"\n";
    }
    // Close the file
    MyFile.close();
    return 0;
}