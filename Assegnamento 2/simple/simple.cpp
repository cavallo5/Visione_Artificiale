/*
Visione Artificiale
Assegnamento 2
*/

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <iterator>
#include <cstdlib>
using namespace std;
using namespace cv;

struct ArgumentList {
	std::string image_name;		    //!< image file name
};


bool ParseInputs(ArgumentList& args, int argc, char **argv) {

	if(argc<3 || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help"))
	{
		std::cout<<"usage: simple -i <image_name>"<<std::endl;
		std::cout<<"exit:  type q"<<std::endl<<std::endl;
		std::cout<<"Allowed options:"<<std::endl<<
				"   -h	                     produce help message"<<std::endl<<
				"   -i arg                   image name. Use %0xd format for multiple images."<<std::endl;
		return false;
	}

	int i = 1;
	while(i<argc)
	{
		if(std::string(argv[i]) == "-i") {
			args.image_name = std::string(argv[++i]);
		}
		++i;
	}

	return true;
}

template <class T>
T bilinearAt(const cv::Mat& image, float r, float c)
{
    float s = c - std::floor(c);
    float t = r - std::floor(r);

    int floor_r = (int) std::floor(r);
    int floor_c = (int) std::floor(c);

    int ceil_r = (int) std::ceil(r);
    int ceil_c = (int) std::ceil(c);

    T value = (1 - s) * (1 - t) * image.at<T>(floor_r, floor_c)
                + s * (1 - t) * image.at<T>(floor_r, ceil_c)
                + (1 - s) * t * image.at<T>(ceil_r, floor_c)
                + s * t * image.at<T>(ceil_r, ceil_c);

    return value;
}



///////////////////////////////////////////////////////////////////////////////////////
//Assegnamento 1
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
///////////////////////////////////////////////////////////////////////////////////////
/*
Funzione che effettua la convoluzione float (versione modificata senza padding)
Input:
	-cv::Mat& image: singolo canale uint8
	-cv::Mat& kernel: singolo canale float32
	-cv::Mat& out: singolo canale float32
	-int i: variabile di servizio
*/
void convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& output, int i) {
  int pad_riga; //variabile utile per gestire il padding 
  int pad_col; //variabile utile per gestire il padding
  output = cv::Mat(image.rows, image.cols, CV_32FC1, cv::Scalar(0)); //definisco dimensione e formato dell'immagine d'uscita
  int valutation_kernel[2]; //array in cui salverò le informazioni per il padding del kernel
  valutazione_kernel(kernel,valutation_kernel); //valuto la dimensione del kernel
  pad_riga=valutation_kernel[0]; //salvo l'informazione riguardanti il padding delle righe in pad_riga
  pad_col=valutation_kernel[1]; //salvo l'informazione riguardanti il padding delle colonne in pad_col
  float * out = (float *) output.data; //puntatori a float per output.data di cv::Mat output
  float * ker = (float *) kernel.data; //puntatori a float per kernel.data di cv::Mat kernel
  if(i==0){
  	 for(int righe = ((pad_riga)/2); righe < image.rows-((pad_riga)/2); righe++)
	{
		for(int colonne = ((pad_col)/2);  colonne < image.cols-((pad_col)/2); colonne++)
		{
			for(int kernel_righe = 0; kernel_righe < kernel.rows; kernel_righe++)
			{
				for(int kernel_colonne = 0; kernel_colonne < kernel.cols; kernel_colonne++)
				{
					out[colonne+righe*output.cols] += (float) image.data[((colonne-((pad_col)/2)+kernel_colonne)+(righe-((pad_riga)/2)+kernel_righe)*image.cols)*image.elemSize()] * ker[(kernel_colonne+kernel_righe*kernel.cols)];
				}}}}}
  if(i==1){
  	float* in = (float*) image.data;
  	for(int righe = ((pad_riga)/2); righe < image.rows-((pad_riga)/2); righe++)
	{
		for(int colonne = ((pad_col)/2);  colonne < image.cols-((pad_col)/2); colonne++)
		{
			for(int kernel_righe = 0; kernel_righe < kernel.rows; kernel_righe++)
			{
				for(int kernel_colonne = 0; kernel_colonne < kernel.cols; kernel_colonne++)
				{
					out[colonne+righe*output.cols] += (float) in[((colonne-((pad_col)/2)+kernel_colonne)+(righe-((pad_riga)/2)+kernel_righe)*image.cols)] * ker[(kernel_colonne+kernel_righe*kernel.cols)];
				}}}}}
}
///////////////////////////////////////////////////////////////////////////////////////
/*
Funzione che permette di effettuare l'operazione di trasposizione  
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
Funzione che genera un kernel Gaussiano orizzontale
Input:
	-float sigma: deviazione standard della gaussiana
	-int radius: raggio del kernel
Output:
	-cv::Mat Kernel 
*/
cv::Mat gaussianKernel(float sigma, int radius) {
  float dimensione_kernel=(2*radius)+1; //dimensione del Kernel ottenuta dal radius
  Mat Kernel = cv::Mat (dimensione_kernel, dimensione_kernel , CV_32FC1); //matrice Kernel singolo canale float 32
  float sum = 0; //variabile che conta la somma dei valori assunti dall'esponenziale nel ciclo per poi normalizzare 
  float* ker = (float*) Kernel.data; //puntatore a float per Kernel.data
  for(int kernel_righe = 0; kernel_righe < Kernel.rows; kernel_righe++)
	{
		for(int kernel_colonne = 0; kernel_colonne < Kernel.cols; kernel_colonne++)
		{
			float esponenziale = (1/(2 * M_PI * pow(sigma, 2)))*exp(-((pow((kernel_colonne-((Kernel.cols-1)/2)), 2) + pow((kernel_righe-((Kernel.rows-1)/2)), 2))/(2 * pow(sigma, 2))));
			ker[(kernel_colonne+kernel_righe*Kernel.cols)] = (float)(esponenziale);
			sum = sum+ker[(kernel_colonne+kernel_righe*Kernel.cols)];
		}
	}
   return Kernel; //restituisco cv::Mat Kernel
}



///////////////////////////////////////////////////////////////////////////////////////
//Assegnamento 2
///////////////////////////////////////////////////////////////////////////////////////
/*
Parte 1
Funzione che calcola i corner di Harris
Input:
	-[in] image: immagine di ingresso, singolo canale uint8
	-[in] alpha: parametro per il calcolo della response 𝜃
	-[in] harrisTh: minima response per avere un corner
	-[out] keypoints: lista dei corner individuati, esperessi come (riga,colonna)
*/
void harrisCornerDetector(const cv::Mat image, std::vector<cv::KeyPoint> & keypoints0, float alpha, float harrisTh)
{
/**********************************
 *
 * PLACE YOUR CODE HERE
 *
 *
 *
 * E' ovviamente viatato utilizzare un detector di OpenCv....
 *
 */

	//Variabili utilizzate
	cv::Mat KernelIx;
	cv::Mat KernelIy;
	cv::Mat Gauss_Ker;
	cv::Mat Ix;
	cv::Mat Iy;
	cv::Mat IxIx;
	cv::Mat IyIy;
	cv::Mat IxIy;
	cv::Mat GIxIx;
	cv::Mat GIyIy;
	cv::Mat GIxIy;
	cv::Mat teta;
	float kerD[3] = {-1,0,1};
	float P,V1,V2,V3,V4,V5,V6,V7,V8;


	//Passo 1: Image derivates
	//Componente x
	KernelIx = cv::Mat(1, 3, CV_32FC1, kerD); //KernelIx=[-1,0,1]
	//Calcolo Ix
	convFloat(image, KernelIx, Ix, 0);
	//Componente y
	trasposta_matrice(KernelIx, KernelIy); //KernelIy=[-1,0,1]'
	//Calcolo Iy
	convFloat(image, KernelIy, Iy, 0);
	
	//Passo 2: Square of derivates
	//Calcolo Ix^2
	IxIx = Ix.mul(Ix);
	//Calcolo Iy^2
	IyIy = Iy.mul(Iy);
	//Calcolo Ix*Iy
	IxIy = Ix.mul(Iy);
	
	//Passo 3: Gaussian 
	Gauss_Ker=gaussianKernel(1,2); //Genero un kernel Gaussiano
	//Calcolo G(Ix^2)
	convFloat(IxIx, Gauss_Ker, GIxIx, 1);
	//Calcolo G(Iy^2)
	convFloat(IyIy, Gauss_Ker, GIyIy, 1);
	//Calcolo G(IxIy)
	convFloat(IxIy, Gauss_Ker, GIxIy, 1);
	
	//Passo 4: Cornerness function
	teta = (GIxIx.mul(GIyIy)) - GIxIy.mul(GIxIy)- alpha*((GIxIx + GIyIy).mul((GIxIx + GIyIy)));

	//Visualizzazione response 𝜃 
	cv::Mat adjMap;
	cv::Mat falseColorsMap;
	double minr,maxr;
	cv::minMaxLoc(teta, &minr, &maxr);
	cv::convertScaleAbs(teta, adjMap, 255 / (maxr-minr));
	cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);
	cv::namedWindow("response1", cv::WINDOW_NORMAL);
	cv::imshow("response1", falseColorsMap);

	//Passo 5: Perform non-maximum suppression
	float* teta_data = (float*) teta.data; //puntatore a float per teta.data
	for(int r = 1; r < teta.rows-1; r++)
	{
		for(int c = 1; c < teta.cols-1; c++)
		{
			//Per effettuare la Non-maximu suppression, devo analizzare il vicinato
			P=teta_data[ (r * teta.cols + c )];
			V1=teta_data[((r-1)*teta.cols+(c-1))]; //r-1,c-1
			V2=teta_data[((r-1)*teta.cols+(c))]; //r-1,c
			V3=teta_data[((r-1)*teta.cols+(c+1))]; //r-1,c+1
			V4=teta_data[((r)*teta.cols+(c-1))]; //r,c-1
			V5=teta_data[((r)*teta.cols+(c+1))]; //r,c+1
			V6=teta_data[((r+1)*teta.cols+(c-1))]; //r+1,c-1
			V7=teta_data[((r+1)*teta.cols+(c))]; //r+1,c
			V8=teta_data[((r+1)*teta.cols+(c+1))]; //r+1,c+1
			//Ora elimino tutti i punti 𝜃 ≤ h𝑎𝑟𝑟𝑖𝑠𝑇h e tutti i punti 𝜃 > h𝑎𝑟𝑟𝑖𝑠𝑇h che non sono massimi locali rispetto al loro vicinato
			if(P > harrisTh && P>V1 && P>V2 && P>V3 && P>V4 && P>V5 && P>V6 && P>V7 && P>V8){
				keypoints0.push_back(cv::KeyPoint(float(c), float(r), 5)); //I restanti punti sono i keypoint
			}		
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////
/*
Parte 2
Funzione che calcola la migliore omografia possibile a partire dai match ottenuti.
Input:
	-[in] points1 e point0: lista di corner che sono stati associati tra le due immagini: points0[i] <-> point1[i]
	-[in] N: numero di iterazioni di RANSAC
	-[in] epsilon: errore massimo di un inlier
	-[in] sample_size: dimensione dei campioni di RANSAC
	-[out] H: omografia
	-[out] inliers_best0, inliers_best1: lista dei corner che risultano essere inliers rispetto ad H
*/
void findHomographyRansac(const std::vector<cv::Point2f> & points1, const std::vector<cv::Point2f> & points0, int N, float epsilon, int sample_size, cv::Mat & H, std::vector<cv::Point2f> & inliers_best0, std::vector<cv::Point2f> & inliers_best1)
{
	/**********************************
	 *
	 * PLACE YOUR CODE HERE
	 *
	 *
	 *
	 *
	 * E' possibile utilizzare:
	 * 		cv::findHomography(sample1, sample0, 0)
	 *
	 * E' vietato utilizzare:
	 * 		cv::findHomography(sample1, sample0, CV_RANSAC)
	 *
	 */
	
	int contatore_bestinliers = 0; //variabile che conserva l'informazione del set di inliers più numeroso
	//sample_size=4;
	std::vector<cv::Point2f> sample0(sample_size); //Inlier matches
	std::vector<cv::Point2f> sample1(sample_size); //Inlier matches
	int contatore_inliers; //variabile conta gli inliers di ogni set
	for (int k = 0; k < N; k++) //numero di iterazioni di RANSAC (100 volte)
	{
		contatore_inliers = 0; //ad ogni iterazione va rimesso a 0 per ricalcolare gli inliers di quella iterazione
		std::vector<cv::Point2f> inliers0;	//vettore per il calcolo di inliers_best0 
		std::vector<cv::Point2f> inliers1;	//vettore per il calcolo di inliers_best1
		//Selezione di 4 match a caso tra quelli (𝑝𝑖0, 𝑝𝑖1) in input:
		for (int i = 0; i < sample_size; i++) //sample_size=4
		{
			 int casuale = rand() % points0.size(); //selezione casuale
			 sample0[i] = points0[casuale]; //riempio sample0
			 sample1[i] = points1[casuale]; //riempio sample1
		}
		//Calcolo H usando i punti trovati 
		H = cv::findHomography(cv::Mat(sample1), cv::Mat(sample0), 0); //Homography approssimata
		double* homography = (double*) H.data; //puntatore a double per H.data
		for (unsigned int j = 0; j < points0.size(); j++)
		{	
			//Uso delle coordinate omogenee 
			double c_omogeneaX = homography[(0 * H.cols + 0)]*points1[j].x + homography[(0 * H.cols + 1)]*points1[j].y + homography[(0 * H.cols + 2)];
			double c_omogeneaY = homography[(1 * H.cols + 0)]*points1[j].x + homography[(1 * H.cols + 1)]*points1[j].y + homography[(1 * H.cols + 2)];
			double c_omogeneaZ = homography[(2 * H.cols + 0)]*points1[j].x + homography[(2 * H.cols + 1)]*points1[j].y + homography[(2 * H.cols + 2)];
			//Ritorno alle coordinate euclidee
			double c_euclideaX = c_omogeneaX / c_omogeneaZ;
			double c_euclideaY = c_omogeneaY / c_omogeneaZ;
			cv::Point2f omographPoint(c_euclideaX,c_euclideaY);
			//Analisi dei match (𝑝𝑖0, 𝑝𝑖1) che soddisfano la trasformazione H a meno di un piccolo errore epsilon
			if ((sqrt(pow((points0[j].x - omographPoint.x),2) + pow((points0[j].y - omographPoint.y),2))) < epsilon) //||𝑝𝑖0, H 𝑝𝑖1|| < ε
			{
				contatore_inliers++; //aumento contatore
				inliers0.push_back(points0[j]); //aggiungo ad inliers0
				inliers1.push_back(points1[j]); //aggiungo ad inliers0
			}
		}
		//inliers0 e inliers1 sono i migliori inliers se sono del set più numeroso:
		if (contatore_inliers >= contatore_bestinliers)
		{
				contatore_bestinliers = contatore_inliers; //uso contatore_bestinliers per salvare il numero più grande del set di inliers
				inliers_best0.clear(); //rimuovo gli elementi precedenti del vettore inliers_best0 dato che ne ho trovato uno più numeroso
			 	inliers_best0 = inliers0; //salvo il nuovo inliers_best0
				inliers_best1.clear(); //rimuovo gli elementi precedenti del vettore inliers_best1 dato che ne ho trovato uno più numeroso
				inliers_best1 = inliers1; //salvo il nuovo inliers_best1
		}
	}
	//Modello di omografia migliore con gli inliers_best0 e inliers_best1 trovati
	H = cv::findHomography(cv::Mat(inliers_best1), cv::Mat(inliers_best0), 0);
}
int main(int argc, char **argv)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;

	//vettore delle immagini di input
	std::vector<cv::Mat> imageRGB_v;
	//vettore delle immagini di input grey scale
	std::vector<cv::Mat> image_v;

	std::cout<<"Simple image stitching program."<<std::endl;

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

		cv::Mat im = cv::imread(frame_name);
		if(im.empty())
		{
			break;
		}

		//opening file
		std::cout<<"Opening "<<frame_name<<std::endl;

		//save RGB image
		imageRGB_v.push_back(im);

		//save grey scale image for processing
		cv::Mat im_grey(im.rows, im.cols, CV_8UC1);
		cvtColor(im, im_grey, CV_RGB2GRAY);
		image_v.push_back(im_grey);

		frame_number++;
	}

	if(image_v.size()<2)
	{
		std::cout<<"At least 2 images are required. Exiting."<<std::endl;
		return 1;
	}

	int image_width = image_v[0].cols;
	int image_height = image_v[0].rows;

	////////////////////////////////////////////////////////
	/// HARRIS CORNER
	//
	float alpha = 0.04;
	float harrisTh = 45000000;    //da impostare in base alla propria implementazione

	std::vector<cv::KeyPoint> keypoints0, keypoints1;

	harrisCornerDetector(image_v[0], keypoints0, alpha, harrisTh);
	harrisCornerDetector(image_v[1], keypoints1, alpha, harrisTh);
	////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////
    /// CALCOLO DESCRITTORI E MATCHES
	//
    int briThreshl=30;
    int briOctaves = 3;
    int briPatternScales = 1.0;
	cv::Mat descriptors0, descriptors1;

	//dichiariamo un estrattore di features di tipo BRISK
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::BRISK::create(briThreshl, briOctaves, briPatternScales);
    //calcoliamo il descrittore di ogni keypoint
    extractor->compute(image_v[0], keypoints0, descriptors0);
    extractor->compute(image_v[1], keypoints1, descriptors1);

    //associamo i descrittori tra me due immagini
    std::vector<std::vector<cv::DMatch> > matches;
	cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING);
	matcher.radiusMatch(descriptors0, descriptors1, matches, image_v[0].cols*0.2);

    std::vector<cv::Point2f> points[2];
    for(unsigned int i=0; i<matches.size(); ++i)
      {
        if(!matches.at(i).empty())
          {
                points[0].push_back(keypoints0.at(matches.at(i).at(0).queryIdx).pt);
                points[1].push_back(keypoints1.at(matches.at(i).at(0).trainIdx).pt);
          }
      }
	////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////
    // CALCOLO OMOGRAFIA
    //
    //
    // E' obbligatorio implementare RANSAC.
    //
    // Per testare i corner di Harris inizialmente potete utilizzare findHomography di opencv, che include gia' RANSAC
    //
    // Una volta che avete verificato che i corner funzionano, passate alla vostra implementazione di RANSAC
    //
    //
    cv::Mat H;            //omografia finale
	std::vector<cv::Point2f> inliers_best[2]; //inliers
    if(points[1].size()>=4)
    {
    	int N=100;            //numero di iterazioni di RANSAC
    	float epsilon = 0.5;  //distanza per il calcolo degli inliers
    	int sample_size = 4;  //dimensione del sample

    	//
    	//
    	// Abilitate questa funzione una volta che quella di opencv funziona
    	//
    	//
    	findHomographyRansac(points[1], points[0], N, epsilon, sample_size, H, inliers_best[0], inliers_best[1]);
    	//
    	//
    	//
    	//
    	//

    	std::cout<<std::endl<<"Risultati Ransac: "<<std::endl;
    	std::cout<<"Num inliers / match totali "<<inliers_best[0].size()<<" / "<<points[0].size()<<std::endl;

    	//
    	//
    	// Rimuovere questa chiamata solo dopo aver verificato che i vostri corner di Harris generano una omografia corretta
    	//
    	//
    	//H = cv::findHomography( cv::Mat(points[1]), cv::Mat(points[0]), CV_RANSAC );
    	//
    	//
    	//
    	//
    	//
    }
    else
    {
    	std::cout<<"Non abbastanza matches per calcolare H!"<<std::endl;
    	H = (cv::Mat_<double>(3, 3 )<< 1.0, 0.0, 0.0,
    		                           0.0, 1.0, 0.0,
			                           0.0, 0.0, 1.0);
    }

    std::cout<<"H"<<std::endl<<H<<std::endl;
    cv::Mat H_inv = H.inv();///H.at<double>(2,2);
    std::cout<<"H_inverse "<<std::endl<<H_inv<<std::endl;
	////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////
    /// CALCOLO DELLA DIMENSIONE DELL'IMMAGINE FINALE
    //
    cv::Mat p = (cv::Mat_<double>(3, 1) << 0, 0, 1);
    cv::Mat tl = H*p;
    tl/=tl.at<double>(2,0);
    p = (cv::Mat_<double>(3, 1) << image_width-1, image_height-1, 1);
    cv::Mat br = H*p;
    br/=br.at<double>(2,0);
    p = (cv::Mat_<double>(3, 1) << 0, image_height-1, 1);
    cv::Mat bl = H*p;
    bl/=bl.at<double>(2,0);
    p = (cv::Mat_<double>(3, 1) << image_width-1, 0, 1);
    cv::Mat tr = H*p;
    tr/=tr.at<double>(2,0);

    int min_warped_r = std::min(std::min(tl.at<double>(1,0), bl.at<double>(1,0)),std::min(tr.at<double>(1,0), br.at<double>(1,0)));
    int min_warped_c = std::min(std::min(tl.at<double>(0,0), bl.at<double>(0,0)),std::min(tr.at<double>(0,0), br.at<double>(0,0)));

    int max_warped_r = std::max(std::max(tl.at<double>(1,0), bl.at<double>(1,0)),std::max(tr.at<double>(1,0), br.at<double>(1,0)));
    int max_warped_c = std::max(std::max(tl.at<double>(0,0), bl.at<double>(0,0)),std::max(tr.at<double>(0,0), br.at<double>(0,0)));

    int min_final_r = std::min(min_warped_r,0);
    int min_final_c = std::min(min_warped_c,0);

    int max_final_r = std::max(max_warped_r,image_height-1);
    int max_final_c = std::max(max_warped_c,image_width-1);

    int width_final = max_final_c-min_final_c+1;
    int height_final = max_final_r-min_final_r+1;

    std::cout<<"width_final "<<width_final<<" height_final "<<height_final<<std::endl;
	////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////
    /// CALCOLO IMMAGINE FINALE
    //
    cv::Mat outwarp(height_final, width_final, CV_8UC3, cv::Scalar(0,0,0));

    //copio l'immagine 0 sul nuovo piano immagine, e' solo uno shift
    imageRGB_v[0].copyTo(outwarp(cv::Rect(std::max(0,-min_warped_c), std::max(0,-min_warped_r), image_width, image_height)));

    //copio l'immagine 1 nel piano finale
    //in questo caso uso la trasformazione prospettica
    for(int r=0;r<height_final;++r)
    {
        for(int c=0;c<width_final;++c)
        {
        	cv::Mat p = (cv::Mat_<double>(3, 1) << c+std::min(0,min_warped_c), r+std::min(0,min_warped_r), 1);
        	cv::Mat pi = H_inv*p;
        	pi/=pi.at<double>(2,0);

        	if(int(pi.at<double>(1,0))>=1 && int(pi.at<double>(1,0))<image_height-1 && int(pi.at<double>(0,0))>=1 && int(pi.at<double>(0,0))<image_width-1)
        	{
        		cv::Vec3b pick = bilinearAt<cv::Vec3b>(imageRGB_v[1], pi.at<double>(1,0), pi.at<double>(0,0));

        		//media
        		if(outwarp.at<cv::Vec3b>(r,c) != cv::Vec3b(0.0))
        			outwarp.at<cv::Vec3b>(r,c) =  (outwarp.at<cv::Vec3b>(r,c)*0.5 + pick*0.5);
        		else
        			outwarp.at<cv::Vec3b>(r,c) = pick;
        	}
        }
    }
	////////////////////////////////////////////////////////

	////////////////////////////
	//WINDOWS
	//
    for(unsigned int i = 0;i<keypoints0.size();++i)
    	cv::circle(imageRGB_v[0], cv::Point(keypoints0[i].pt.x , keypoints0[i].pt.y ), 5,  cv::Scalar(0), 2, 8, 0 );

    for(unsigned int i = 0;i<keypoints1.size();++i)
    	cv::circle(imageRGB_v[1], cv::Point(keypoints1[i].pt.x , keypoints1[i].pt.y ), 5,  cv::Scalar(0), 2, 8, 0 );

	cv::namedWindow("KeyPoints0", cv::WINDOW_AUTOSIZE);
	cv::imshow("KeyPoints0", imageRGB_v[0]);

	cv::namedWindow("KeyPoints1", cv::WINDOW_AUTOSIZE);
	cv::imshow("KeyPoints1", imageRGB_v[1]);

    cv::Mat matchsOutput(image_height, image_width*2, CV_8UC3);
    imageRGB_v[0].copyTo(matchsOutput(cv::Rect(0, 0, image_width, image_height)));
    imageRGB_v[1].copyTo(matchsOutput(cv::Rect(image_width, 0, image_width, image_height)));
    for(unsigned int i=0; i<points[0].size(); ++i)
    {
    	cv::Point2f p2shift = points[1][i];
    	p2shift.x+=imageRGB_v[0].cols;
    	cv::circle(matchsOutput, points[0][i], 3, cv::Scalar(0,0,255));
    	cv::circle(matchsOutput, p2shift, 3, cv::Scalar(0,0,255));
    	cv::line(matchsOutput, points[0][i], p2shift, cv::Scalar(255,0,0));
    }
	cv::namedWindow("Matches", cv::WINDOW_NORMAL);
	cv::imshow("Matches", matchsOutput);

    cv::Mat matchsOutputIn(image_height, image_width*2, CV_8UC3);
    imageRGB_v[0].copyTo(matchsOutputIn(cv::Rect(0, 0, image_width, image_height)));
    imageRGB_v[1].copyTo(matchsOutputIn(cv::Rect(image_width, 0, image_width, image_height)));
    for(unsigned int i=0; i<inliers_best[0].size(); ++i)
    {
    	cv::Point2f p2shift = inliers_best[1][i];
    	p2shift.x+=image_width;
    	cv::circle(matchsOutputIn, inliers_best[0][i], 3, cv::Scalar(0,0,255));
    	cv::circle(matchsOutputIn, p2shift, 3, cv::Scalar(0,0,255));
    	cv::line(matchsOutputIn, inliers_best[0][i], p2shift, cv::Scalar(255,0,0));
    }
	cv::namedWindow("Matches Inliers", cv::WINDOW_NORMAL);
	cv::imshow("Matches Inliers", matchsOutputIn);

	cv::namedWindow("Outwarp", cv::WINDOW_AUTOSIZE);
	cv::imshow("Outwarp", outwarp);

	cv::waitKey(0);
	////////////////////////////

	return 0;
}
