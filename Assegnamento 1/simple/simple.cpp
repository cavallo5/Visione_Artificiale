/*
Visione Artificiale
Assegnamento 1
*/

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;

struct ArgumentList {
	std::string image_name;		//!< image file name
	int wait_t;                     //!< waiting time
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {

if(argc<3 || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help"))
	{
		std::cout<<"usage: simple -i <image_name>"<<std::endl;
		std::cout<<"exit:  type q"<<std::endl<<std::endl;
		std::cout<<"Allowed options:"<<std::endl<<
				"   -h	                     produce help message"<<std::endl<<
				"   -i arg                   image name. Use %0xd format for multiple images."<<std::endl<<
				"   -t arg                   wait before next frame (ms) [default = 0]"<<std::endl<<std::endl<<std::endl;
		return false;
	}

	int i = 1;
	while(i<argc)
	{
		if(std::string(argv[i]) == "-i") {
			args.image_name = std::string(argv[++i]);
		}

		if(std::string(argv[i]) == "-t") {
			args.wait_t = atoi(argv[++i]);
		}
		else
			args.wait_t = 0;

		++i;
	}

	return true;
}
///////////////////////////////////////////////////////////////////////////////////////
/*
Esercizio 1
Funzione che effettua la valutazione del kernel
Input:
	-cv::Mat& kernel
	-int val[2]
La funzione salverà:
	-il padding della riga in val[0]
	-il padding della colonna in val[1]
*/
void valutazione_kernel(const cv::Mat& kernel, int val[])
{
	if (kernel.rows > kernel.cols) //se righe del kernel > colonne del kernel
  	{
    	val[0] = kernel.rows - 1;
    	val[1] = 0;
  	}  
  	else if (kernel.cols > kernel.rows) //se colonne del kernel > righe del kernel
  	{
  		val[0] = 0; 
    	val[1] = kernel.cols - 1;
  	}
  	else //kernel quadrato
  	{
  		val[0] = kernel.rows - 1;
    	val[1] = kernel.cols - 1;
  	}	
}

/*
Esercizio 1
Funzione che effettua il padding 
Input:
	cv::Mat& input: immagine su cui fare il padding 
	cv::Mat& output: immagine in cui salvo l'immagine con il padding eseguito
	int pad_riga: informazioni sul padding delle righe
	int pad_colonna: informazione sul padding delle colonne
*/
void padding_immagine (const cv::Mat& input, cv::Mat& output,int pad_riga, int pad_colonna){
  output = cv::Mat::zeros(input.rows + pad_riga, input.cols + pad_colonna, CV_8UC1); //creo un output di zeri con le dimensioni e formato richieste
  for(int i = 0; i < input.rows; i++)
  {
    for(int j = 0 ; j < input.cols ; j++ ) 
    {
      output.data[( j + pad_colonna/2 + (i + pad_riga/2) * output.cols) * output.elemSize()] = input.data[(j + i * input.cols) * input.elemSize()]; //padding
    }
  }
}
/*
Esercizio 1
Funzione che effettua la convoluzione float
Input:
	-cv::Mat& image: singolo canale uint8
	-cv::Mat& kernel: singolo canale float32
	-cv::Mat& out: singolo canale float32
*/
void convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& output) {
  int pad_riga; //variabile utile per gestire il padding 
  int pad_col; //variabile utile per gestire il padding
  float somma; //variabile utile per effettuare la convoluzione
  output = Mat(image.rows, image.cols, CV_32FC1); //definisco dimensione e formato dell'immagine d'uscita
  //Valuto il kernel
  int valutation_kernel[2]; //array in cui salverò le informazioni per il padding del kernel
  valutazione_kernel(kernel,valutation_kernel); //valuto la dimensione del kernel
  pad_riga=valutation_kernel[0]; //salvo l'informazione riguardanti il padding delle righe in pad_riga
  pad_col=valutation_kernel[1]; //salvo l'informazione riguardanti il padding delle colonne in pad_col
  /* Richiamo la funzione per effettuare il padding dell'immagine */
  cv::Mat immagine_con_padding; //definisco una cv::Mat in cui salverò l'immagine con il padding eseguito
  padding_immagine(image, immagine_con_padding, pad_riga, pad_col); //effettuo il padding 
  /* Puntatori a float per serializzare matrice */
  float * out = (float *) output.data; //puntatori a float per output.data di cv::Mat output
  float * ker = (float *) kernel.data; //puntatori a float per kernel.data di cv::Mat kernel
  for (int righe = pad_riga/2; righe < immagine_con_padding.rows - pad_riga/2 ; righe++) 
  {
    for (int colonne = pad_col/2 ; colonne < immagine_con_padding.cols - pad_col/2; colonne++)
    {
      somma = 0; //inizializzo somma a 0 prima di fare la convoluzione
      for (int kernel_righe = 0; kernel_righe < kernel.rows; kernel_righe++)
      {
        for(int kernel_colonne = 0; kernel_colonne < kernel.cols; kernel_colonne++)
        {
          somma = somma + float(immagine_con_padding.data[((righe - pad_riga/2 + kernel_righe ) * immagine_con_padding.cols  + ( colonne - pad_col/2 + kernel_colonne)) * immagine_con_padding.elemSize()]) * ker[kernel_righe * kernel.cols + kernel_colonne]; //convoluzione
         }
      }
      out[(righe - pad_riga/2) * output.cols  + (colonne - pad_col/2) ] = somma; //salvo in out i valori di somma ottenuti dalla convoluzione
    }
  }
  	//Effettuo una conversione di output in float 32
	cv::Mat conversione= Mat(image.rows, image.cols, CV_32FC1, out).clone(); //creo matrice di conversione delle dimensioni richieste
	conversione.convertTo(output, CV_32FC1); //converto output in float 32
}
/* 
Esercizio 1bis
Funzione che effettua la convoluzione int
Input:
	-cv::Mat& image: singolo canale uint8
	-cv::Mat& kernel: singolo canale float32
	-cv::Mat& out: singolo canale uint8
*/
void conv(const cv::Mat& image, cv::Mat& kernel, cv::Mat& output){
  int pad_riga; //variabile utile per gestire il padding 
  int pad_col; //variabile utile per gestire il padding
  float somma; //variabile utile per effettuare la convoluzione
  const float massimo  = numeric_limits<float>::max();
  const float minimo  = numeric_limits<float>::min();
  output = Mat(image.rows, image.cols, CV_8UC1); //definisco dimensione e formato dell'immagine d'uscita
  //Valuto ik kernel
  int valutation_kernel[2]; //array in cui salverò le informazioni per il padding del kernel
  valutazione_kernel(kernel,valutation_kernel); //valuto la dimensione del kernel
  pad_riga=valutation_kernel[0]; //salvo l'informazione riguardanti il padding delle righe in pad_riga
  pad_col=valutation_kernel[1]; //salvo l'informazione riguardanti il padding delle colonne in pad_col
  /* Richiamo la funzione per effettuare il padding dell'immagine */	
  cv::Mat immagine_con_padding; //definisco una cv::Mat in cui salverò l'immagine con il padding eseguito
  padding_immagine (image, immagine_con_padding, pad_riga, pad_col); //effettuo il padding 
  float max = minimo;
  float min = massimo;
  float * out = new float[image.rows * image.cols]; 
  float * ker = (float*) kernel.data; //puntatori a float per kernel.data di cv::Mat kernel
  for (int righe = pad_riga/2; righe < immagine_con_padding.rows - pad_riga/2 ; righe++) 
  {
    for (int colonne = pad_col/2 ; colonne < immagine_con_padding.cols - pad_col/2; colonne++)
    {
      somma = 0; //inizializzo somma a 0 prima di fare la convoluzione
      for (int kernel_righe = 0; kernel_righe < kernel.rows; kernel_righe++)
      {
        for(int kernel_colonne = 0; kernel_colonne < kernel.cols; kernel_colonne++)
        {
          somma = somma + float(immagine_con_padding.data [((righe - pad_riga/2 + kernel_righe ) * immagine_con_padding.cols  + ( colonne - pad_col/2 + kernel_colonne)) * immagine_con_padding.elemSize()]) * ker[kernel_righe * kernel.cols + kernel_colonne]; //convoluzione
         }
      }
      //Aggiorno i valori di massimo e minimo per effettuare successivamente la constrast stretching
      if (somma > max ) 
          max = somma;
      else if(somma < min)
          min = somma;
      out[ (righe - pad_riga/2) * image.cols + (colonne -pad_col/2)  ] = somma; //salvo in out i valori di somma ottenuti dalla convoluzione
    }
  }
  // Riscalatura dei valori nel range [0-255]
  cv::Mat conversione = Mat(image.rows, image.cols, CV_32FC1, out).clone(); //creo matrice di conversione delle dimensioni richieste
  conversione= conversione- min;
  float divisore=max - min;
  conversione = conversione * (255/divisore); //range [0-255]
  conversione.convertTo(output, CV_8UC1); //converto output in uint8
}
///////////////////////////////////////////////////////////////////////////////////////
/*
Esercizio 2
Funzione che genera un kernel di blur Gaussiano 1-D orizzontale
Input:
	-float sigma: deviazione standard della gaussiana
	-int radius: raggio del kernel
Output:
	-cv::Mat Kernel 
*/
cv::Mat gaussianKernel(float sigma, int radius) {
  float dimensione_kernel=(2*radius)+1; //dimensione del Kernel ottenuta dal radius
  Mat Kernel = cv::Mat (1, 2*radius + 1 , CV_32FC1); //matrice Kernel singolo canale float 32
  float sum = 0; //variabile che conta la somma dei valori assunti dall'esponenziale nel ciclo per poi normalizzare 
  float* ker = (float*) Kernel.data; //puntatore a float per Kernel.data
  for (int i = 0; i < dimensione_kernel; i++){
      float esponenziale= exp(-0.5 * ((i-radius)/sigma) * ((i-radius)/sigma)); //Gaussian blur
      ker[i]=float(esponenziale); //salvo i valori nel kernel 
	    sum = sum+(float)esponenziale; //aggiorno il valore somma 
    }
   Kernel=Kernel/sum; //normalizzo dividendo per la somma dei pesi
   return Kernel; //restituisco cv::Mat Kernel
}
///////////////////////////////////////////////////////////////////////////////////////
/*
Esercizio 3
Funzione che permette di ottenere un Gaussian blur verticale trasponendo quello orizzontale
Input:
	-cv::Mat& matrice
Output:
	-cv::Mat& matrice_trasposta
*/
void trasposta_matrice(const cv::Mat& matrice, cv::Mat& matrice_trasposta)
{
  matrice_trasposta=Mat(matrice.cols,1,CV_32FC1); //matrice trasposta singolo canale float 32
  float * mat = (float *) matrice.data; //puntatore a float della matrice iniziale
  float * mat_tra = (float *) matrice_trasposta.data; //puntatore a float di matrice_trasposta
  for(int i=0; i < matrice.cols; i++) 
  {
    for(int j=0; j < matrice.rows; j++)
    {
      mat_tra[j + i*matrice_trasposta.cols ] = mat[i + j*matrice.cols]; //traspongo gli elementi della matrice matrice
    }
  }
}
///////////////////////////////////////////////////////////////////////////////////////
/*
Esercizio 5
Funzione che permette di calcolare la magnitudo e l'orientazione di Sobel 3x3
Input:
	-cv::Mat& image: immagine 
	-cv::Mat& magnitude: magnitudo di Sobel
	-cv::Mat& orientation: orientazione di Sobel
*/
void sobel(const cv::Mat& image, cv::Mat& magnitude, cv::Mat& orientation){
		const float massimo = std::numeric_limits<float>::max();
		const float minimo = std::numeric_limits<float>::min();
		cv::Mat G_orizzontale = (Mat_<float>(3,3) << 1, 0, -1, +2, 0, -2 , 1, 0, -1); //G_orizzontale=[1,0,-1;2,0,-2;1,0,-1]
		cv::Mat G_verticale = (Mat_<float>(3,3) << 1, 2, 1, 0, 0, 0 , -1, -2, -1); //G_verticale=[1,2,1;0,0,0;-1,-2,-1]
		cv::Mat G_orizzontale_x , G_verticale_y; 
		convFloat ( image, G_orizzontale , G_orizzontale_x );
		convFloat ( image,G_verticale ,G_verticale_y );
		float min = massimo;
		float max = minimo;
		float* magnitudo = (float* ) magnitude.data; //puntatore a float di magnitude
		float* orientazione = (float* ) orientation.data; //puntatore a float di orientation
		float* D_gx = (float* ) G_orizzontale_x.data; //puntatore a float di G_orizzontale_x
		float* D_gy = (float* ) G_verticale_y.data; //puntatore a float di G_verticale_y
		for (int i = 0; i < image.rows; i++){ 
			for (int j = 0; j < image.cols; j++){
				//Calcolo l'orientazione
				orientazione [ (i * orientation.cols + j) ] = atan2 (D_gy [ (i * G_verticale_y.cols + j)] , D_gx [ (i * G_orizzontale_x.cols + j) ] ); //arctg(Gy/Gx)
				if ( orientazione [ (i * orientation.cols + j) ] < 0)  //se ho valori negativi 
					orientazione [ (i * orientation.cols + j) ] = orientazione [ (i * orientation.cols + j) ] + 2*M_PI; //sommo 2*M_PI per rifasare
				//Calcolo la magnitudo
				magnitudo[(i * magnitude.cols + j)] = sqrt((pow( D_gx [(i * G_orizzontale_x.cols + j)] ,2) + pow( D_gy [ (i * G_verticale_y.cols+ j) ],2) ) ); //sqrt(Gx^2 + Gy^2)
				 //Aggiorno i valori di max e min per effettuare successivamente la riscalatura
				if( magnitudo [(i * magnitude.cols + j)]< min)
				{
					min = magnitudo [(i * magnitude.cols+j )];
				}	
				else if( magnitudo [(i * magnitude.cols + j )] > max )
				{
					max = magnitudo [ (i * magnitude.cols + j)];
				}
			}
		}
	  	// Riscalatura dei valori nel range [0-255]
		Mat conversione = Mat(image.rows,image.cols,CV_32FC1,magnitudo).clone(); //creo matrice di conversione con le dimensioni e formato richiesti
		conversione =conversione - min;
  		float divisore=max - min;
  		conversione = conversione * (1.0/divisore);
		conversione.convertTo(magnitude,CV_32FC1); ////converto output in float32
}
///////////////////////////////////////////////////////////////////////////////////////
/*
Esercizio 6
Funzione che permette di calcolare l'interpolazione bilineare
Input:
	-cv::Mat& image: immagine uint8
	-float r= riga del punto
	-float c= colonna del punto
Output:
	-float f=valore dell'interpolazione bilineare
*/
float bilinear(const cv::Mat& image, float r, float c)
{
   	float f=0; //variabile in cui salverò il valore della bilinear
    int x = int(r); //x è la parte intera di r
    int y = int(c); //y è la parte intera di c
    float s = r - x; 
    float t = c - y; 
    float f00 =( image.data[(x*image.cols +  y)]); //in(y,x) 
    float f10 = (image.data[((x+1) * image.cols + y)]); //in(y,x+1)
    float f01 = (image.data[(x * image.cols + (y  + 1))]); //in(y+1,x)
    float f11 =( image.data[((x+1) * image.cols + (y+1))]); //in(y+1,x+1)
	f=(1-s)*(1-t)* f00+ s*(1-t)* f10 + (1-s)*t* f01+ s*t*f11; //formula slide 15
	return f;
} 
///////////////////////////////////////////////////////////////////////////////////////
/*
Esercizio 7
Funzione che permette di calcolare l'interpolazione bilineare
Input:
	-cv::Mat& image: immagine float 32
	-float r= riga del punto
	-float c= colonna del punto
Output:
	-float f=valore dell'interpolazione bilineare
*/
float bilinear_float(const Mat& image, float r, float c)
{ 

  	float f=0;
    int x = floor(r); //x è l'intero arrotondato per difetto
    int y = floor(c); //y è l'intero arrotondato per difetto
    float s = r - x; 
    float t = c - y; 
	  float* immagine = (float * ) image.data;
    float f00 =float ( immagine[(  x  * image.cols +  y ) ] ); //in(y,x) 
    float f10 = float (immagine[ ( ( x+1) * image.cols + y )] ); //in(y,x+1)
    float f01 = float ( immagine[( x * image.cols + (y  + 1))] ); //in(y+1,x)
    float f11 =(immagine[( ( x+1) * image.cols + (y  + 1))] ); //in(y+1,x+1)
    f=(1-s)*(1-t)* f00+ s*(1-t)* f10 + (1-s)*t* f01+ s*t*f11; //formula slide 15
    return f;
} 
/*
Esercizio 7
Funzione che permette di calcolare trovare i picchi (massimi) del gradiente nella direzione perpendicolare al gradiente stesso
Input:
	-cv::Mat& magnitude: singolo canale float32
	-cv::Mat& orientation: singolo canale float32
	-cv::Mat& out: singolo canale float 32
	-float th
*/
void findPeaks (const cv::Mat& magnitude, const cv::Mat& orientation, cv::Mat& out ,float th){
	out = Mat( magnitude.rows, magnitude.cols, CV_32FC1); //creo un output con le dimensioni e formato richieste (float32)
	float E_1X,E_1Y,E_1,E_2,E_2X,E_2Y; //variabili utilizzate per calcolare i vicini a distanza 1 lungo la direzione del gradiente
	float* magnitudo = (float*) magnitude.data; //puntatore a float di magnitude
	float* uscita = (float*) out.data; //puntatore a float di out
	float* orientazione = (float*) orientation.data; //puntatore a float di orientation
	for (int r = 0; r < magnitude.rows; r++){
		for (int c = 0; c < magnitude.cols; c++){
			E_1X = c + 1 * cos( orientazione[ r * out.cols + c ]); //e1x=c+1*cos(theta)
			E_1Y = r + 1 * sin( orientazione[ r * out.cols + c ]); //e1y=r+1*sin(theta)
			E_1 = bilinear_float(magnitude,E_1Y,E_1X); //e1 ottenuta dalla bilinear di E_1X e E_1Y
			E_2X = c - 1 * cos( orientazione[ r * out.cols + c ]); //e2x=c-1*cos(theta)
			E_2Y = r - 1 * sin( orientazione[ r * out.cols + c ] ); //e2y=r-1*sin(theta)
			E_2 = bilinear_float(magnitude,E_2Y, E_2X); //e2 ottenuta dalla bilinear di E_2X e E_2Y
			//Soppressione dei non massimi
			if ( magnitudo[ (r * magnitude.cols + c )] >= E_1 && magnitudo[ (r * magnitude.cols + c )] >= E_2 && magnitudo[ (r * magnitude.cols + c )]  >= th ) 
			{
				uscita[ r * out.cols + c ] = magnitudo[ (r * magnitude.cols + c )]; //out(r,c)=in(r,c)
			}    
			else
			{
				 uscita[ r * out.cols + c ]  = 0.0f; //out(r,c)=0
			}
		}
	}
	//Effettuo la conversione in float32
	cv::Mat conversione = Mat(magnitude.rows,magnitude.cols,CV_32FC1,uscita).clone();  //creo matrice di conversione con le dimensioni e formato richiesti
	conversione.convertTo(out,CV_32FC1); 
}
///////////////////////////////////////////////////////////////////////////////////////
/*
Esercizio 8
Funzione che realizza la Soglia con isteresi
	-cv::Mat& magnitude: singolo canale float32
	-cv::Mat& out: singolo canale uint8
	-float th1
	-float th2
*/
void doubleTh(Mat& magnitude,Mat& out, float th1, float th2){
	out = Mat(magnitude.rows, magnitude.cols, CV_8UC1); //definisco l'output con le dimensioni e il formato richieste (uint8)
	const float val_massimo=255;
	const float val_medio=128;
	const float val_minimo=0;
	float* magnitudo = (float* ) magnitude.data; //puntatore a float di magnitude
	for (int riga = 0; riga < magnitude.rows; riga++){		
		for (int colonna = 0; colonna < magnitude.cols; colonna++){
			//Catena di if che realizzano il sistema di slide 19
			if ( magnitudo [ riga * magnitude.cols + colonna ] > th1 ) //in(r,c)>th1
			{
				out.data[ (  riga * out.cols + colonna ) * out.elemSize()] = val_massimo; //out(r,c)=255
				
			}
			else if ( magnitudo [ riga * magnitude.cols + colonna ]  <= th1 && magnitudo [ riga * magnitude.cols + colonna ] > th2 ) //th1>=in(r,c)>th2
			{
				out.data[ (  riga * out.cols + colonna ) * out.elemSize()] = val_medio; //out(r,c)=128
			}
			else //in(r,c)<=th2
			{
				out.data[ (  riga * out.cols + colonna ) * out.elemSize()] = val_minimo; //out(r,c)=0
			}
		}
	}	
}
///////////////////////////////////////////////////////////////////////////////////////
/*
Esercizio 9
Funzione che effettua il Canny Edge Detector su lenna.pgm
	-cv::Mat& image: singolo canale uint8
	-cv::Mat& out: singolo canale uint8
	-float th
	-float th1
	-float th2
*/
void canny(const Mat& image, Mat& out, float th, float th1, float th2)
{
	out = Mat(image.rows,image.cols,CV_8UC1); //definisco l'output con le dimensioni e il formato richieste (uint8)
	//1. Sobel magnituo e orientazione
	cv::Mat magnitudo,orientazione;
	magnitudo = Mat(image.rows, image.cols, CV_32FC1);
	orientazione =  Mat(image.rows, image.cols, CV_32FC1);
	sobel(image,magnitudo,orientazione);
	//2. FindPeaks della magnitudo
	cv::Mat uscita_findPeaks;
	findPeaks(magnitudo, orientazione, uscita_findPeaks, th);
	//3. Sogliatura con isteresi
	doubleTh(uscita_findPeaks,out,th1, th2);
}
///////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;
	/*
	Valori soglia assegnamento
	a=0.2;
	b=0.7;
	c=0.3;
	*/
	float a,b,c; 
	cout << "Valore di th: ";
	cin >> a;
	cout << "Valore di th_1: ";
	cin >> b;
	cout << "Valore di th_2: ";
	cin >> c;
	//////////////////////
	//parse argument list:
	//////////////////////
	ArgumentList args;
	if(!ParseInputs(args, argc, argv)) {
		return 1;
	}
	while(!exit_loop)
	{
		//generating file name
		//
		//multi frame case
		if(args.image_name.find('%') != std::string::npos)
			sprintf(frame_name,(const char*)(args.image_name.c_str()),frame_number);
		else //single frame case
			sprintf(frame_name,"%s",args.image_name.c_str());
		//opening file
		std::cout<<"Opening "<<frame_name<<std::endl;
		cv::Mat image = cv::imread(frame_name,IMREAD_GRAYSCALE);
		if(image.empty())
		{
			std::cout<<"Unable to open "<<frame_name<<std::endl;
			return 1;
		}
		//////////////////////
		//processing code here
		///////////////////////////////////////////////////////////////////////////////////////
		//Esercizio 1
		//Creo Kernel 3x3 float 32 con elementi pari a 1
		cv::Mat kernel(3,3,CV_32FC1); //Creo un kernel singolo canale float32 con tutti gli elementi pari a 1
		for(int riga_kernel=0; riga_kernel < kernel.rows; riga_kernel++)
		{
			for(int colonna_kernel=0; colonna_kernel < kernel.cols; colonna_kernel++)
			{
				*((float *)&kernel.data[(colonna_kernel+riga_kernel*kernel.cols)*kernel.elemSize()])=1.0;
				
			}
		}
		//Stampa del Kernel
		cout<<"Il kernel è:"<<endl;
		cout<<kernel<<endl;
		cv::Mat uscita_esercizio_1; //Definisco cv::Mat esercizio_1
		conv(image,kernel,uscita_esercizio_1);
		//Mostro i risultati ottenuti
		namedWindow("Esercizio_1", WINDOW_NORMAL);
		imshow("Esercizio_1", uscita_esercizio_1);
		///////////////////////////////////////////////////////////////////////////////////////
		//Esercizio 2
		cout<<"Creazione kernel di blur Gaussiano 1-D orizzontale"<<endl;
		float sigma=1.0; //sigma della gaussiana
		int raggio=3; //raggio del kernel
		cout<<"Sigma: "<<sigma<<endl;
		cout<<"Raggio: "<<raggio<<endl;
		cv::Mat gaus_ker=gaussianKernel(sigma, raggio); //Genero un kernel di blur Gaussiano
		cout<<"Il kernel di blur Gaussiano 1-D orizzontale è"<<std::endl;
		cout<<gaus_ker<<endl;
		///////////////////////////////////////////////////////////////////////////////////////
		//Esercizio 3
		cout<<"Generazione Gaussian blur orizzontale"<<endl;
		cv::Mat gaussian_blur_orizzontale; //Gaussian blur orizzontale
		conv(image, gaus_ker, gaussian_blur_orizzontale);
		cout<<"Generazione Gaussian blur verticale"<<endl;
	  cv::Mat gaus_ker_trasp; //kernel verticale
	  trasposta_matrice(gaus_ker, gaus_ker_trasp); //ottengo il kernel verticale trasponendo quello orizzontale
	  cv::Mat gaussian_blur_verticale; //Gaussian blur verticale
		conv(image, gaus_ker_trasp, gaussian_blur_verticale);
		cout<<"Generazione Gaussian blur bidimensionale"<<endl;
		cv::Mat gaussian_blur_bid; //Gaussian blur bidimensionale
		conv(gaussian_blur_orizzontale, gaus_ker_trasp, gaussian_blur_bid);
		//Mostro i 3 risultati ottenuti
		imshow("Gaussian blur orizzontale", gaussian_blur_orizzontale);
    imshow("Gaussian blur verticale",gaussian_blur_verticale);
    imshow("Gaussian blur bidimensionale", gaussian_blur_bid);
		///////////////////////////////////////////////////////////////////////////////////////
		//Esercizio 4
		cv::Mat monodimensionale = (Mat_<float>(1,3) << -1, 0, 1); //derivativo monodimensionale [-1,0,1]
		cv::Mat laplace= (Mat_<float>(3,3) << 0, 1, 0, 1, -4, 1, 0, 1, 0); //[0,1,0;1,-4,1;0,1,0]
		cv::Mat gauss_derivativo_orizzontale= gaus_ker_trasp * monodimensionale; //derivativo Gaussiano orizzontale
		cv::Mat monodimensionale_trasp; //ci salvo il trasposto del derivativo monodimensionale ovvero  [-1,0,1]'
		trasposta_matrice(monodimensionale, monodimensionale_trasp); //effettuo il trasposto del derivativo monodimensionale
		cv::Mat gauss_derivativo_verticale= monodimensionale_trasp * gaus_ker; //derivativo Gaussiano verticale
		cout<<"Generazione filtro derivativo Gaussiano orizzontale"<<endl;
		cv::Mat filtro_derivativo_orizzontale; //ci salvo il filtro derivativo gaussiano orizzontale
		conv(image,gauss_derivativo_orizzontale,filtro_derivativo_orizzontale);
		cout<<"Generazione filtro derivativo Gaussiano verticale"<<endl;		
		Mat filtro_derivativo_verticale; //ci salvo il filtro derivativo gaussiano verticale
		conv(image,gauss_derivativo_verticale,filtro_derivativo_verticale);			
		cout<<"Generazione filtro Laplaciano"<<endl;		
		cv::Mat filtro_laplaciano; //ci salvo il filtro Laplaciano
		conv(image,laplace,filtro_laplaciano);
		//Mostro i 3 risultati ottenuti
		imshow("Derivativo Gaussiano Orizzontale", filtro_derivativo_orizzontale);
    imshow("Derivativo Gaussiano Verticale", filtro_derivativo_verticale);
    imshow("Laplaciano",filtro_laplaciano);
		///////////////////////////////////////////////////////////////////////////////////////
		//Esercizio 5
		cv::Mat magnitudo, orientazione; //magnitudo e orientazione
    magnitudo = Mat(image.rows, image.cols, CV_32FC1); //creo magnitudo con le dimensioni e formato richieste (float32)
    orientazione =  Mat(image.rows, image.cols, CV_32FC1); //creo orientazione con le dimensioni e formato richieste (float32)
    cout<<"Generazione magnitudo e orientazione di Sobel"<<endl;		
    sobel(image,magnitudo,orientazione);
    //Visualizzazione dell'orientazione di Sobel
    cv::Mat adjMap;
		convertScaleAbs(orientazione, adjMap, 255 / (2*M_PI));
		cv::Mat falseColorsMap;
		applyColorMap(adjMap, falseColorsMap,cv::COLORMAP_AUTUMN);
		imshow("Orientazione (Sobel) ", falseColorsMap);
		//Visualizzazione della magnitudo di Sobel
		imshow("Magnitudo (Sobel)", magnitudo);
		///////////////////////////////////////////////////////////////////////////////////////
		//Esercizio 6
		float r_ex=27.8; //r dell'esempio
		float c_ex=11.4; //c dell'esempio
		float bilinear_interpolation=bilinear(image,r_ex,c_ex); //calcolo dell'interpolazione bilineare
		cout<<"Valore di interpolazione bilineare:"<<endl;
		cout<<bilinear_interpolation<<endl;
		///////////////////////////////////////////////////////////////////////////////////////
		//Esercizio 7
		cv::Mat findPeaks_uscita; //ci salvo Find Peaks
		float th=a; //0.2
		cout<<"Generazione Find Peaks of Edge Responses"<<endl;
		findPeaks (magnitudo,orientazione,findPeaks_uscita,th); 
		//Visualizzazione di findPeaks
		imshow ("Find Peaks", findPeaks_uscita);
		///////////////////////////////////////////////////////////////////////////////////////
		//Esercizio 8
		float th1= b; //0.7
		float th2=c; //0.3
		cv::Mat soglia_isteresi; 
		cout<<"Generazione Soglia con isteresi"<<endl;
		doubleTh(magnitudo, soglia_isteresi, th1, th2);
		imshow("Soglia con Isteresi", soglia_isteresi);
		///////////////////////////////////////////////////////////////////////////////////////
		//Esercizio 9
		cv::Mat canny_edge; //salvo l'uscita del Canny Edge Detector
		cout<<"Canny Edge Detector"<<endl;
		canny(image,canny_edge,th,th1,th2);
		//Visualizzazione di Canny Edge Detector
		imshow("Canny Edge Detector", canny_edge);
		///////////////////////////////////////////////////////////////////////////////////////

		//display image
		cv::namedWindow("image", cv::WINDOW_NORMAL);
		cv::imshow("image", image);

		//wait for key or timeout
		unsigned char key = cv::waitKey(args.wait_t);
		std::cout<<"key "<<int(key)<<std::endl;
		//char key='g';
		//here you can implement some looping logic using key value:
		// - pause
		// - stop
		// - step back
		// - step forward
		// - loop on the same frame
		if(key == 'g')
			exit_loop = true;

		frame_number++;
	}

	return 0;
}
