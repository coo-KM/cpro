#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nn.h"

void fc(int m, int n, const float * x, const float * A, const float * b, float * y){
    /*ｍxｎ行列A,ｍ行の列ベクトルｂ,ｎ行の列ベクトルｘに対して yk = ak*x + bk (k = 1,...,m)によりｍ行の列ベクトルｙを書き込む*/
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
    //入力値xが負の場合0正の場合入力値をyで出力する関数
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

void softmaxwithloss_bwd(int n, const float * y, unsigned char t, float * dEdx){
    //y:softmaxによる出力 t:正解の添字を表す
    //softmax層に向かう勾配　dEdx　を求める
    for(int i = 0;i < n;i++){
        if(i == t){
            dEdx[i] = y[i] - 1.0;
        }
        else{
            dEdx[i] = y[i];
        }
    }
}

void relu_bwd(int n, const float * x, const float * dEdy, float * dEdx){
    //x:順伝搬におけるrelu層の入力 dEdy:上流側からの勾配　
    //relu層に向かう勾配　dEdx　を求める
    for(int i = 0;i < n;i++){
        if(x[i] > 0){
           dEdx[i] = dEdy[i]; 
        }
        else{
            dEdx[i] = 0;
        }
    }
}

void fc_bwd(int m, int n, const float * x, const float * dEdy, const float * A, float * dEdA, float * dEdb, float * dEdx){
    int h,i,j,k;
    //x:順伝搬におけるfc層での入力　dedy:上流層からの勾配
    //Aに向かう勾配　dEdA,　b に向かう勾配　dEdb　を求める
    for(h = 0;h < m;h++){
        for(i = 0;i < n;i++){
            dEdA[h * n + i] = dEdy[h] * x[i];
        }
        dEdb[h] = dEdy[h];
    }
    //fc層に向かう勾配　dEdx　を求める
    for(j = 0;j < n;j++){
        dEdx[j] = 0;
        for(k = 0;k < m;k++){
            dEdx[j] = A[n*k+j] * dEdy[k] + dEdx[j];
        }
    }
}

void shuffle(int n, int * x) {
    //n個の配列の順序をランダムに入れ替える
    int i,j,k;
    
    for(i = 0; i < n; i++){
        x[i] = i;
    }
    for(i = 0;i < n; i++){
        j = rand() % n;
        k = x[i];
        x[i] = x[j];
        x[j] = k;
    }
} 

float cross_entropy_error(const float * y, int t) { 
    //損失関数の計算
    return -1*log(y[t] + 1e-7);
}

void add(int n, const float * x, float * o) { 
    // o[i] += x[i] を実行 
    for(int i = 0 ;i < n ;i++ ){
        o[i] += x[i];
    }
} 

void scale(int n, float x, float * o) { 
    // o[i] *= x を実行 
    for(int i = 0 ;i < n ;i++ ){
        o[i] *= x;
    }
} 

void init(int n, float x, float * o) { 
    // o[i] = x を実行 
    for(int i = 0 ;i < n ;i++ ){
        o[i] = x;
    }
} 

void shuffle_2(int n, float * x) {
    //n個の配列の順序をランダムに入れ替える。rand_symmetry やboxmuller で使用
    int i,j,k;

    for(i = 0;i < n; i++){
        j = rand() % n;
        k = x[i];
        x[i] = x[j];
        x[j] = k;
    }
} 

void rand_symmetry(int n,float *o){
    //[-1:1]の乱数の生成ただし average_x[i] = 0 （原点に関して対称）を満たす。
    //今回この乱数は用いなかった
    float rand_1 ;
    if(n % 2 ==0){
        for(int i = 0; i < n ; i += 2){
            //[0:1] の乱数
            rand_1 = (double)rand() / (double)RAND_MAX ;

            //原点対称を満たすように負の数も導入
            o[i] = rand_1;
            o[i + 1] = -1 * rand_1;
        }
    }else{
        for(int i = 0; i < n-1 ; i += 2){
            //[0:1] の乱数
            rand_1 = (double)rand() / (double)RAND_MAX ;

            //原点対称を満たすように負の数も導入
            o[i] = rand_1;
            o[i + 1] = -1 * rand_1;
        }
        o[n-1] =  0.0;
    }
    //順序がそろっていた配列 o をランダムな順序にする
    shuffle_2(n, o);
}

void boxmuller(float * A ,int n , const float disp , const float ave){
    //disp:分散　 ave：平均　を満たすガウス分布の乱数
    //A 初期化したい行列
    //boxmuller法によりガウス分布とする。今回この関数による乱数も用いた
    const float pi = 3.14159;
    float rand_1 = 0.0;
    float rand_2 = 0.0;
    float Gau_1 = 0.0;
    float Gau_2 = 0.0;
    //ガウス分布に直す乱数の要素数が偶数個の場合
    if(n % 2 == 0){
        for(int i = 0; i < n ; i += 2){
            //[0:1] の乱数
            rand_1 = (double)rand() / ((double)RAND_MAX + 1.0);
            rand_2 = (double)rand() / ((double)RAND_MAX + 1.0);

            //boxmuller法
            Gau_1 = ave + sqrt(disp) * sqrt(-2.0 * log(rand_1)) * cos(2 * pi * rand_2) ;
            Gau_2 = ave + sqrt(disp) * sqrt(-2.0 * log(rand_1)) * sin(2 * pi * rand_2) ;

            A[i] = Gau_1;
            A[i + 1] = Gau_2;
        }
    }else{
        //奇数個の場合
        for(int i = 0; i < n-1 ; i += 2){
            //[0:1] の乱数。            
            rand_1 = (double)rand() / ((double)RAND_MAX + 1.0);
            rand_2 = (double)rand() / ((double)RAND_MAX + 1.0);
            //boxmuller法　
            Gau_1 = ave + sqrt(disp) * sqrt(-2.0 * log(rand_1)) * cos(2 * pi * rand_2) ;
            Gau_2 = ave + sqrt(disp) * sqrt(-2.0 * log(rand_1)) * sin(2 * pi * rand_2) ;
    
            A[i] =  Gau_1;
            A[i+1] =  Gau_2;
        }
        //[0:1] の乱数。            
        rand_1 = (double)rand() / ((double)RAND_MAX + 1.0);
        rand_2 = (double)rand() / ((double)RAND_MAX + 1.0);
        //boxmuller法　
        Gau_1 = ave + sqrt(disp) * sqrt(-2.0 * log(rand_1)) * cos(2 * pi * rand_2) ;
        
        A[n-1] = Gau_1;
    }
    //順序がそろってガウス分布となっていた A をランダムな順所にする
    shuffle_2(n, A);
}

void rand_init(int n,float *o){
    //[-1:1]の乱数の生成　ただし今回用いなかった
    for(int i=0;i<n;i++){
        o[i] = (double)(rand() - (RAND_MAX/2) ) / (RAND_MAX/2);
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
    
    //最大のｙの添字をを求める
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
    //メモリの開放
    free(y_FC1);
    free(y_RelU1);
    free(y_FC2);
    free(y_Relu2);
    free(y_FC3);
    
    //y_maxの添え字を返す
    return subscript_max;
}

void backward6(const float * A1, const float * b1, const float * A2, const float * b2, const float * A3, const float * b3, const float * x, unsigned char t, float * y, float * dEdA1, float * dEdb1, float * dEdA2, float * dEdb2, float * dEdA3, float * dEdb3) {
    //6層の順伝搬　→　逆伝搬によりそれぞれの層, A, bに向かう勾配を求める
    //A,b: fc層（1，2，3）で用いる行列 x:入力　t:正解データ
    //メモリの確保
    float *y_FC1 = malloc(sizeof(float) * 50);
    float *y_FC2 = malloc(sizeof(float) * 100);
    float *y_FC3 = malloc(sizeof(float) * 10);
    float *y_RelU1 = malloc(sizeof(float) * 50);
    float *y_Relu2 =malloc(sizeof(float) * 100);

    //順伝搬
    //FC1層 入力は　ｘ
    fc(50 ,784, x , A1 , b1 , y_FC1);
    //Relu1層　入力は　y_FC1
    relu(50, y_FC1, y_RelU1);
    
    //FC2層　入力は y_Relu1
    fc(100, 50, y_RelU1 , A2,  b2, y_FC2);
    //Relu2層　入力は　y_FC2
    relu( 100, y_FC2, y_Relu2);
    
    //FC3層　入力は　y_Relu2
    fc(10, 100, y_Relu2 , A3, b3, y_FC3);
    //softmax層　入力は　y_FC3
    softmax(10, y_FC3, y);

    //逆伝搬
    //最後のsoftmax層からFC3層に向かう勾配
    float * dEdx_softmax = malloc(sizeof(float)*10);
    softmaxwithloss_bwd(10,y,t,dEdx_softmax);

    //FC3層からA3、ｂ3、relu2層に向かう勾配
    float * dEdx_fc3 = malloc(sizeof(float)*100);
    fc_bwd(10, 100 , y_Relu2,dEdx_softmax, A3 , dEdA3, dEdb3,dEdx_fc3);
    
    //relu2層からFC2層に向かう勾配
    float * dEdx_relu2 = malloc(sizeof(float)*100);
    relu_bwd(100 ,y_FC2, dEdx_fc3 ,dEdx_relu2);

    //FC2層からA2、ｂ2、relu1層に向かう勾配
    float * dEdx_fc2 = malloc(sizeof(float)*50);    
    fc_bwd(100, 50 , y_RelU1, dEdx_relu2, A2 , dEdA2, dEdb2, dEdx_fc2);

    //relu1層からFC1に向かう勾配
    float * dEdx_relu1 = malloc(sizeof(float)*50);
    relu_bwd(50 ,y_FC1, dEdx_fc2 ,dEdx_relu1);

    //FC1層からA1、ｂ1、に向かう勾配
    float * dEdx_fc1 = malloc(sizeof(float)*784);    
    fc_bwd(50, 784 , x , dEdx_relu1, A1 , dEdA1, dEdb1, dEdx_fc1);

    //メモリの開放
    free(dEdx_softmax);
    free(dEdx_fc1);
    free(dEdx_relu1);
    free(dEdx_fc2);
    free(dEdx_relu2);
    free(dEdx_fc3);
    free(y_FC1);
    free(y_RelU1);
    free(y_FC2);
    free(y_Relu2);
    free(y_FC3);
}

void save(const char * filename, int m, int n, const float *A, const float * b){
    //行数ｍ列数ｎの行列Aと列ベクトルｂを受け取り保存する関数
    FILE *fp;
    fp = fopen(filename, "wb");
    fwrite(A, sizeof(float), m * n, fp);
    fwrite(b, sizeof(float), m, fp);
    fclose(fp);
}

int main(int argc, char * argv[]){
    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;

    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;

    int width = -1;
    int height = -1;

    load_mnist(&train_x, &train_y, &train_count,&test_x, &test_y, &test_count,&width, &height);

    // これ以降，３層NN の係数 A_784x10 および b_784x10 と，
    // 訓練データ train_x + 784*i (i=0,...,train_count-1), train_y[0]～train_y[train_count-1],
    // テストデータ test_x + 784*i (i=0,...,test_count-1), test_y[0]～test_y[test_count-1],
    // を使用することができる．

    srand(time(NULL));

    //エポック回数
    int epoch;
    epoch = 20;
    
    //ミニバッチサイズ
    int n;
    n = 10;
    
    //学習率
    float lerning_rate;
    lerning_rate = 0.1;

    //それぞれのメモリの確保
    float * A1 = malloc(sizeof(float)*784*50);
    float * b1 = malloc(sizeof(float)*50);
    float * A2 = malloc(sizeof(float)*50*100);
    float * b2 = malloc(sizeof(float)*100);
    float * A3 = malloc(sizeof(float)*100*10);
    float * b3 = malloc(sizeof(float)*10);

    float * dEdA1_ave = malloc(sizeof(float)*784*50);
    float * dEdb1_ave = malloc(sizeof(float)*50);
    float * dEdA2_ave = malloc(sizeof(float)*50*100);
    float * dEdb2_ave = malloc(sizeof(float)*100);
    float * dEdA3_ave = malloc(sizeof(float)*100*10);
    float * dEdb3_ave = malloc(sizeof(float)*10);

    float * dEdA1 = malloc(sizeof(float)*784*50); 
    float * dEdb1 = malloc(sizeof(float)*50);
    float * dEdA2 = malloc(sizeof(float)*50*100); 
    float * dEdb2 = malloc(sizeof(float)*100); 
    float * dEdA3 = malloc(sizeof(float)*100*10);
    float * dEdb3 = malloc(sizeof(float)*10); 

    float * y = malloc(sizeof(float)*10);
 
    //A,b をガウス分布の乱数でリセット
    boxmuller(A1, 784*50, sqrt(2.0 / 784.0 ),0.0);
    boxmuller(b1, 50, sqrt(2.0 / 784.0 ),0.0);
    boxmuller(A2, 50*100, sqrt(2.0 / 50.0),0.0);
    boxmuller(b2, 100, sqrt(2.0 / 50.0 ),0.0);
    boxmuller(A3, 100*10, sqrt(2.0 / 100.0),0.0);
    boxmuller(b3, 10, sqrt(2.0 / 100.0),0.0);
    
    //index導入
    int * index = malloc(sizeof(int)*train_count);

    //以下エポック回数分くり返す
    for(int i = 0; i < epoch; i++){
        shuffle(train_count , index);
    
        //ミニバッチ学習を　train_count / n　回行う
        int k = 0;

        for(int j = 0; j < train_count / n; j++){
            //dEdA_ave,dEdb_aveを0でリセット
            init(784 * 50 , 0, dEdA1_ave);
            init(50 , 0, dEdb1_ave);
            init(50 * 100 , 0, dEdA2_ave);
            init(100 , 0, dEdb2_ave);
            init(100 * 10 , 0, dEdA3_ave);
            init(10 , 0, dEdb3_ave);


            //インデックスからn個取り出してそれぞれについてdEdA,dEdbを求める
            for(; k < n * (j + 1);k++ ){
                backward6(A1,b1,A2,b2,A3,b3,train_x+784*index[k],train_y[index[k]],y,dEdA1,dEdb1,dEdA2,dEdb2,dEdA3,dEdb3);
                
                //dEdA1_aveにdEdA1の値を足す
                add(784 * 50, dEdA1 , dEdA1_ave);
                //dEdb1_aveにdEdb1の値を足す
                add(50, dEdb1 , dEdb1_ave);

                //dEdA2_aveにdEdA2の値を足す
                add(50 * 100, dEdA2 , dEdA2_ave);
                //dEdb2_aveにdEdb2の値を足す
                add(100, dEdb2 , dEdb2_ave);                
                
                //dEdA3_aveにdEdA3の値を足す
                add(100 * 10, dEdA3 , dEdA3_ave);
                //dEdb3_aveにdEdb3の値を足す
                add(10, dEdb3 , dEdb3_ave);
            }
            
            //dEdA_ave,dEdb_aveを　-lerning_rate / n　倍とする
            scale(784 * 50,-lerning_rate / n, dEdA1_ave);
            scale(50, -lerning_rate / n, dEdb1_ave);
            //A＝A + dEdA_ave , b＝b + dEdb_ave とする
            add(784*50,dEdA1_ave,A1);
            add(50,dEdb1_ave,b1);
            
            scale(100 * 50,-lerning_rate / n, dEdA2_ave);
            scale(100, -lerning_rate / n, dEdb2_ave);
            add(100*50,dEdA2_ave,A2);
            add(100,dEdb2_ave,b2);
            
            scale(100 * 10,-lerning_rate / n, dEdA3_ave);
            scale(10, -lerning_rate / n, dEdb3_ave);
            add(100*10,dEdA3_ave,A3);
            add(10,dEdb3_ave,b3);
        }

        float loss = 0;
        float ans = 0;
        for(int i = 0; i < test_count ; i++){
            //正解率
            if(afterward6(A1,b1,A2,b2,A3,b3,test_x + 784 * i ,y) == test_y[i]){
                ans++;
            }
             //損失関数
            loss += cross_entropy_error( y, test_y[i] );
        }
        loss = loss / test_count;
        ans = (ans / test_count) * 100.0;
        printf("epoch %d / 20\n", i+1);
        printf("loss function: %.4f\n", loss);
        printf("answer rate  : %.4f%%\n\n", ans);
    }
    //A,b　を記録
    save(argv[1], 50, 784, A1, b1); 
    save(argv[2], 100, 50, A2, b2); 
    save(argv[3], 10, 100, A3, b3);
    return 0;
}