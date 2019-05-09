#include "iostream"
#include "car.h"
#include "overall_schedule.h"
#include "util.h"

using namespace std;

int main(int argc, char *argv[])
{
    std::cout << "Begin" << std::endl;
	
	if(argc < 6){
		std::cout << "please input args: carPath, roadPath, crossPath, answerPath" << std::endl;
		exit(1);
	}
	
	std::string carPath(argv[1]);
	std::string roadPath(argv[2]);
	std::string crossPath(argv[3]);
	std::string presetAnswerPath(argv[4]);
	std::string answerPath(argv[5]);
	
	std::cout << "carPath is " << carPath << std::endl;
	std::cout << "roadPath is " << roadPath << std::endl;
	std::cout << "crossPath is " << crossPath << std::endl;
	std::cout << "presetAnswerPath is " << presetAnswerPath << std::endl;
	std::cout << "answerPath is " << answerPath << std::endl;
	
	// TODO:read input filebuf
    overall_schedule OS;
    OS.load_cars_roads_crosses(carPath, roadPath, crossPath);
	// TODO:process
    OS.load_preset_answer(presetAnswerPath);
    double start,end,cost;
    start=clock();
    OS.cars_path_regulation();
    int T = OS.schedule_cars(0);
    std::cout << "T = " << T << std::endl;
	// TODO:write output file
    OS.save_answer(answerPath);
    end=clock();
    cost=end-start;
    cout << double(cost)/CLOCKS_PER_SEC << endl;
    /*
    list<int> a;
    a.push_back(1);
    a.push_back(10);
    int* p = &a.back();
    list<int>::iterator iter = a.begin();
    iter ++;
    cout << *iter << endl;
    *p = 2;
    cout << *iter << endl;
    list<int> b = a;
    iter = b.begin();
    iter ++;
    *iter = 3;
    iter = a.begin();
    iter ++;
    cout << *iter << endl;
    b.push_back(20);
    a = b;
    cout << *iter << endl;
    *p = 4;
    cout << *iter << endl;
    */
	
	return 0;
}