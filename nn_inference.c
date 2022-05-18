#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nn.h"

void load(const char * filename, int m, int n, float * A, float * b) {
    //保存したファイルから行数ｍ列数ｎの行列を読み込む関数
    FILE *fp;
    fp = fopen(filename, "rb");
    fread(A, sizeof(float), m * n, fp);
    fread(b, sizeof(float), m, fp);
    fclose(fp);
} 

void fc(int m, int n, const float * x, const float * A, const float * b, float * y){
    /*ｍxｎ行列A,ｍ行の列ベクトルｂ,ｎ行の列ベクトルｘに対して yk = akx + bk (k = 1,...,m)によりｍ行の列ベクトルｙを書き込む*/
    int i, j ;
    for (i = 0; i < m; i++){
        int k = 0;
        y[i] = 0;
        for (j = i * n; j < n * (i + 1); j++){
            y[i] = A[j] * x[k] + y[i];
            k++;
        }
        y[i] = b[i] + y[i];
    }
}

void relu(int n, const float * x, float * y){
    //入力値ｘが負の場合0正の場合入力値を出力する関数
    for (int i = 0; i < n; i++){
        if(x[i] > 0){
            y[i] = x[i];
        }else{
            y[i] = 0;
        }
    }
}

void softmax(int n, const float * x, float * y){
    //n個の入力ｘに対してｘの最大値x_max,exp( x- x_max)の和sumを求め,それぞれの入力x_iに対して y_i =  exp(x_i - x_max) / sum を計算 
    float x_max;
    float sum = 0;
    x_max = x[0];
    for(int i = 0; i < n; i++){
        if(x_max <= x[i]){
            x_max = x[i];
        }
    }
    for (int j = 0;  j < n; j++){
        sum = sum + exp(x[j] - x_max);
    }
    for (int k = 0; k < n;k++){
        y[k] = exp(x[k] - x_max) / sum;
    }
}

int afterward6(const float * A1, const float * b1, const float * A2, const float * b2, const float * A3, const float * b3,const float * x,float * y){
    //順伝搬によりyを求め,yの中の最大値 y_max の添え字を返す
    //A,b fc層（1，2，3）で用いる行列

    //メモリの確保
    float *y_FC1 = malloc(sizeof(float) * 50);
    float *y_FC2 = malloc(sizeof(float) * 100);
    float *y_FC3 = malloc(sizeof(float) * 10);
    float *y_RelU1 = malloc(sizeof(float) * 50);
    float *y_Relu2 =malloc(sizeof(float) * 100);
    //FC1層
    fc(50 ,784, x , A1 , b1 , y_FC1);
    //Relu1層
    
    relu(50, y_FC1, y_RelU1);
    //FC2層
    fc(100, 50, y_RelU1 , A2,  b2, y_FC2);
    //Relu2層
    relu( 100, y_FC2, y_Relu2);
    //FC3層
    fc(10, 100, y_Relu2 , A3, b3, y_FC3);
    softmax(10, y_FC3, y);
    
    //最大のｙの添字を返す
    int j = 0;
    int subscript_max = 0;
    float y_max;
    y_max = y[0];

    for (j = 0; j < 10;j++){
        if (y_max <= y[j]){
            y_max = y[j];
            subscript_max = j;
        }
    }
    free(y_FC1);free(y_RelU1);
    free(y_FC2);free(y_Relu2);
    free(y_FC3);

    return subscript_max;
}

int main(int argc, char * argv[]) {
    float * A1 = malloc(sizeof(float)*784*50); 
    float * b1 = malloc(sizeof(float)*50); 
    float * A2 = malloc(sizeof(float)*50*100); 
    float * b2 = malloc(sizeof(float)*100); 
    float * A3 = malloc(sizeof(float)*100*10); 
    float * b3 = malloc(sizeof(float)*10); 
    float * x = load_mnist_bmp(argv[4]); 
    float * y = malloc(sizeof(float)*10);

    int num_recog;

    load(argv[1], 50, 784, A1, b1); 
    load(argv[2], 100, 50, A2, b2); 
    load(argv[3], 10, 100, A3, b3);

    num_recog = afterward6(A1,b1,A2,b2,A3,b3,x,y);
    printf("%d",num_recog);

    return 0; 
}