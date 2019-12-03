#include <stdio.h>
#include <math.h>
#include <time.h>
#include <bits/stdc++.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <vector>
#include<fstream>

using namespace std;

typedef int city_order;


int city_quantity = 100;
int cross_rand_range = int(city_quantity / 4);
const int population_size = 10000;
const int pop_route_max = 10000000;

double cityTypeDistance[3][3] = {
        {0.10, 0.075, 0.05},
        {0.075, 0.05, 0.025},
        {0.05, 0.025, 0.01}
};

unsigned seed = (unsigned)time(0);

struct city_info
{
    double x;
    double y;
    int type;
};

vector<city_info> city_vector;

// read_file the structure of city from filename
void read_file(const char *filename);

// The dictionary of the city distance
vector<vector<double>> CityDistance;

struct TSP{
    vector<vector<int>> population_route;
    vector<double> pop_fitness;
    vector<double> pop_route_dist;
    vector<int> best_route;

    double BestFitness;
    double BestValue;
    int BestNum;
};

void calc_dist_dictionary();

bool check(TSP &city,int pop,int num,int k);

// initialize the population
void Init(TSP &city);

// calculate the fitness
void CalFitness(TSP &city);

// select algorithm: Roulette
void roulette_select(TSP &city);

// Cross: don't change the head and the tail, change the cities between those nodes
// the probability of cross is probability_cross
void Cross(TSP &city, double probability_cross);

// get a pop_fitness
double cal_fitness(const int *set);
double cal_fitness(vector<int>& set);

// the mutation of population
// the possibility of mutation is possibility_mutation
void Mutation(TSP &city, double possibility_mutation);

void city_print(TSP &city);

void print_iteration(int i, TSP &city);

void print_result(TSP &city);

int main()
{
    /*
/home/tong/Desktop/A3/newGA/tsp100.txt
/home/tong/Desktop/A3/newGA/tsp200.txt
/home/tong/Desktop/A3/newGA/tsp500.txt
     */

    cout << "\nPlease input the name:\n";
    char filename[200];
    cin >> filename;
    read_file(filename);

    for (int city_i = 0; city_i < city_quantity; ++city_i) {
        vector<double> dis(city_quantity, 0.0);
        CityDistance.push_back(dis);
    }

    TSP city;

    for (int pop_iter = 0; pop_iter < population_size; ++pop_iter) {
        if (pop_iter == 0)
            for (int city_iter = 0; city_iter <= city_quantity; ++city_iter) {
                city.best_route.push_back(0.0);
            }
        city.pop_fitness.push_back(0.0);
        city.pop_route_dist.push_back(0.0);
        city.population_route.push_back(vector<int>(city_quantity+1, 0));
    }

//    city_print(city);

    // probability of mutation
    double probability_mutation = 0.05;

    // probability of cross
    double probability_cross = 0.5;
    int epochs = 30000;

    srand(seed);

    calc_dist_dictionary();
    Init(city);
    CalFitness(city);

    int i = 1;
    cout << endl;
    print_iteration(i, city);

    for (; i <= epochs; i++)
    {

        roulette_select(city);
        Cross(city,probability_cross);
        Mutation(city,probability_mutation);
        CalFitness(city);

        if (i % 20 == 0)
            print_iteration(i, city);
    }


    print_result(city);

    return 0;
}


// read_file the structure of city from filename
void read_file(const char *filename)
{
    freopen(filename, "r", stdin);
    city_quantity = 0;
    cin >> city_quantity;  //the number of city,the first data
    cout << city_quantity << endl;
    city_vector.clear();
    for (int i = 0; i < city_quantity; i++) {
        city_info t;
        int c_type;
        cin  >> t.x >> t.y >> c_type;
        t.type = c_type-1;
        city_vector.push_back(t);  // p stored city data
    }
}

double cal_fitness(vector<int>& set)
{
    int city_iter = 0;
    int route_start, route_end;
    double dis = 0;
    while(city_iter < city_quantity)
    {
        route_start = set[city_iter];
        route_end = set[city_iter + 1];
        dis += CityDistance[route_start][route_end];
        city_iter++;
    }

    // get the pop_fitness and return
    double fitness = 1.0 / dis;
    return fitness;
}

double cal_fitness(const int *set)
{
    int city_iter = 0;
    int route_start, route_end;
    double dis = 0;
    while(city_iter < city_quantity)
    {
        route_start = set[city_iter];
        route_end = set[city_iter + 1];
        dis += CityDistance[route_start][route_end];
        city_iter++;
    }

    // get the pop_fitness and return
    double fitness = 1.0 / dis;
    return fitness;
}

void calc_dist_dictionary()
{
    int i,j;
    double temp1, temp2, dis;
    for(i = 0; i < city_quantity; i++){
        for(j = 0;j <= city_quantity; j++){//最后一个城市还应该返回到出发节点
            temp1 = 1.0 * city_vector[j].x - 1.0 * city_vector[i].x;
            temp2 = 1.0 * city_vector[j].y - 1.0 * city_vector[i].y;
            dis = sqrt(temp1 * temp1 + temp2*temp2);
            dis = 1.0 * cityTypeDistance[city_vector[i].type][city_vector[j].type] * dis;
            CityDistance[i][j] = dis;
        }
    }
}

void city_print(TSP &city)
{
    for (int pop_iter = 0; pop_iter < population_size; ++pop_iter) {

        for (int city_iter = 0; city_iter <= city_quantity; ++city_iter) {
            if(pop_iter == 0)
            {
                cout << city.best_route[city_iter] << " ";
//                cout << city.population_route[pop_iter][city_iter] << " ";
            }
        }
        if (pop_iter == 300)
        {
            cout << endl;
            cout << city.pop_fitness[pop_iter];
            cout << city.pop_route_dist[pop_iter];
        }
    }
}

void print_iteration(int i, TSP &city)
{
    cout << "iter "<< i  << ": the optimal cost is : " << city.BestValue << endl;
}

void print_result(TSP &city)
{
    int city_iter;
    // show the best route:
    cout << "\nThe best route is : " << endl;
    for(city_iter = 0; city_iter <= city_quantity; city_iter++)
        printf("%5d", city.best_route[city_iter]);
    // show the minimum route cost:
    printf("\nThe minimum route cost is : %f\n", (city.BestValue));
}

bool check(TSP &city,int pop,int num,int k)
{
    int i;
    for(i=0;i<=num;i++){
        if(k==city.population_route[pop][i])
            return true;
    }
    return false;
}

// initialize the population
void Init(TSP &city)
{

    int city_iter, r;
    int population_iter = 0;
    while (population_iter < population_size)
    {
        city.population_route[population_iter][0]=0;
        city.population_route[population_iter][city_quantity]=0;
        city.BestValue = pop_route_max;
        // pop_fitness is better to be big
        city.BestFitness=0;
        population_iter++;
    }

    population_iter = 0;
    while (population_iter < population_size)
    {
        city_iter = 1;
        while (city_iter < city_quantity)
        {
            r = rand() % (city_quantity - 1) + 1;
            while(check(city, population_iter, city_iter, r))
            {
                r = rand() % (city_quantity - 1 )+ 1;
            }
            city.population_route[population_iter][city_iter] = r;
            city_iter++;
        }
        population_iter++;
    }
}

void CalFitness(TSP &city)
{
    int pop_iter, city_iter;
    int route_start, route_end;
    int best_fitness_pop = 0;
    for(pop_iter = 0; pop_iter < population_size; pop_iter++){
        city.pop_route_dist[pop_iter]=0;
        for(city_iter = 1; city_iter <= city_quantity;city_iter++){
            route_start=city.population_route[pop_iter][city_iter-1];route_end=city.population_route[pop_iter][city_iter];
            city.pop_route_dist[pop_iter]=city.pop_route_dist[pop_iter]+CityDistance[route_start][route_end];
        }
        city.pop_fitness[pop_iter] = 1.0 /city.pop_route_dist[pop_iter];
        if(city.pop_fitness[pop_iter]>city.pop_fitness[best_fitness_pop])
            best_fitness_pop = pop_iter;
    }
    // copy the best rooting to best_route city
    int city_iter_copy=0;
    while (city_iter_copy < city_quantity+1)
    {
        city.best_route[city_iter_copy] = city.population_route[best_fitness_pop][city_iter_copy];
        city_iter_copy++;
    }

    city.BestFitness=city.pop_fitness[best_fitness_pop];
    city.BestValue=city.pop_route_dist[best_fitness_pop];
    city.BestNum=best_fitness_pop;
}

// select algorithm: Roulette
void roulette_select(TSP &city)
{
    vector<vector<int>> tmp_pop(population_size, vector<int>(city_quantity+1, 0));

    int pop_iter;
    int pop_iter2;
    double random_s;

    vector<double> g_population(population_size, 0);

    vector<double> select_population(population_size+1, 0);

    double city_fitness_sum = 0;

    pop_iter = 0;
    while (pop_iter < population_size)
    {
        city_fitness_sum += city.pop_fitness[pop_iter];
        pop_iter++;
    }

    pop_iter = 0;
    while (pop_iter < population_size)
    {
        g_population[pop_iter] = city.pop_fitness[pop_iter]/city_fitness_sum;
        pop_iter++;
    }

    select_population[0] = 0;

    pop_iter=0;
    while (pop_iter < population_size)
    {
        select_population[pop_iter+1] = select_population[pop_iter] + g_population[pop_iter] * RAND_MAX;
        pop_iter++;
    }

    int city_iter = 0;
    while (city_iter < city_quantity+1)
    {
        tmp_pop[0][city_iter] = city.population_route[city.BestNum][city_iter];
        ++city_iter;
    }

    pop_iter2 = 1;
    while (pop_iter2<population_size)
    {
        double ran = rand() % RAND_MAX + 1;
        random_s= (double) ran / 100.0;

        pop_iter = 1;
        while (pop_iter < population_size)
        {
            if(select_population[pop_iter] >= random_s)
                break;
            pop_iter++;
        }

        int city_iter = 0;
        while (city_iter < city_quantity+1)
        {
            tmp_pop[pop_iter2][city_iter] = city.population_route[pop_iter-1][city_iter];
            ++city_iter;
        }

        pop_iter2++;
    }

    pop_iter = 0;
    while (pop_iter < population_size)
    {
        int city_iter = 0;
        while (city_iter < city_quantity+1)
        {
            city.population_route[pop_iter][city_iter] = tmp_pop[pop_iter][city_iter];
            ++city_iter;
        }
        pop_iter++;
    }
}

// Cross: don't change the head and the tail, change the cities between those nodes
// the probability of cross is probability_cross
void Cross(TSP &city, double probability_cross)
{
    vector<city_order> break_order(city_quantity+1, 0);

    vector<city_order> Temp1(city_quantity+1, 0);

    int population_iter,iteration_in_random,city_iter,rand_iter;
    int city_random_iter, best_number_try, random_population_iter;

    for(population_iter=0;population_iter < population_size; population_iter++)
    {
        double random_cross_test = ((double)(rand() % RAND_MAX)) / RAND_MAX;
        if(random_cross_test < probability_cross)
        {
            random_population_iter = rand() % population_size;
            best_number_try = random_population_iter;
            if(best_number_try==city.BestNum||random_population_iter==city.BestNum)
                continue;

            rand_iter=rand()%cross_rand_range+1;  //1-cross_rand_range
            city_random_iter=rand() % (city_quantity-rand_iter)+1; //1-37

            int k = 0;
            while (k <= city_quantity)
            {
                break_order[k] = 0;
                k++;
            }

            iteration_in_random = 1;
            while (iteration_in_random <= rand_iter)
            {
                Temp1[iteration_in_random] = city.population_route[random_population_iter][city_random_iter+iteration_in_random-1]; //city_random_iter+L=2~38 20~38
                break_order[Temp1[iteration_in_random]] = 1;
                iteration_in_random++;
            }

            city_iter=1;
            while (city_iter < city_quantity)
            {
                if(break_order[city.population_route[best_number_try][city_iter]]==0)
                {
                    Temp1[iteration_in_random++]=city.population_route[best_number_try][city_iter];
                    break_order[city.population_route[best_number_try][city_iter]] = 1;
                }
                city_iter++;
            }

            int city_iter2 = 0;
            while (city_iter2 < city_quantity+1)
            {
                city.population_route[best_number_try][city_iter2] = Temp1[city_iter2];
                ++city_iter2;
            }
        }
    }

}

// the mutation of population
// the possibility of mutation is possibility_mutation
void Mutation(TSP &city, double possibility_mutation)
{
    int certain_pop, city_iter;

    vector<int> tmp_population(city_quantity+1);

    int population_iter = 0;
    while(population_iter < population_size)
    {
        double s=((double)(rand()%RAND_MAX))/RAND_MAX;
        certain_pop=rand() % population_size;
        if(s<possibility_mutation&&certain_pop!=city.BestNum)//i!=city.BestNum
        {
            int random_city_number1,random_city_number2,t;
            random_city_number1=(rand()%(city_quantity-1))+1;
            random_city_number2=(rand()%(city_quantity-1))+1;

//            copy(tmp_population,city.population_route[certain_pop]);
            int city_iter_copy=0;
            while (city_iter_copy < city_quantity+1)
            {
                tmp_population[city_iter_copy] = city.population_route[certain_pop][city_iter_copy];
                city_iter_copy++;
            }

            if(random_city_number1>random_city_number2)
            {
                t=random_city_number1;
                random_city_number1=random_city_number2;
                random_city_number2=t;
            }
            for(city_iter=random_city_number1;city_iter<(random_city_number1+random_city_number2)/2;city_iter++)
            {
                t=tmp_population[city_iter];
                tmp_population[city_iter]=tmp_population[random_city_number1+random_city_number2-city_iter];
                tmp_population[random_city_number1+random_city_number2-city_iter]=t;
            }

            if(cal_fitness(tmp_population)< cal_fitness(city.population_route[certain_pop]))
            {
                random_city_number1=(rand() % (city_quantity-1))+1;
                random_city_number2=(rand() % (city_quantity-1))+1;

                //copy(tmp_population,city.population_route[i]);
//                memcpy(tmp_population,city.population_route[certain_pop],sizeof(tmp_population));
                int city_iter_copy=0;
                while (city_iter_copy < city_quantity+1)
                {
                    tmp_population[city_iter_copy] = city.population_route[certain_pop][city_iter_copy];
                    city_iter_copy++;
                }

                if(random_city_number1 > random_city_number2)
                {
                    t=random_city_number1;
                    random_city_number1 = random_city_number2;
                    random_city_number2 = t;
                }

                city_iter=random_city_number1;
                while(city_iter < (random_city_number1 + random_city_number2 ) / 2)
                {
                    t=tmp_population[city_iter];
                    tmp_population[city_iter]=tmp_population[random_city_number1+random_city_number2-city_iter];
                    tmp_population[random_city_number1+random_city_number2-city_iter]=t;
                    city_iter++;
                }

                if(cal_fitness(tmp_population) < cal_fitness(city.population_route[certain_pop]))
                {
                    random_city_number1 = (rand()%(city_quantity-1))+1;
                    random_city_number2 = (rand()%(city_quantity-1))+1;

//                    memcpy(tmp_population,city.population_route[certain_pop],sizeof(tmp_population));
                    int city_iter_copy = 0;
                    while (city_iter_copy < city_quantity+1)
                    {
                        tmp_population[city_iter_copy] = city.population_route[certain_pop][city_iter_copy];
                        city_iter_copy++;
                    }

                    if(random_city_number1>random_city_number2)
                    {
                        t=random_city_number1;
                        random_city_number1=random_city_number2;
                        random_city_number2=t;
                    }
                    city_iter=random_city_number1;
                    while(city_iter < (random_city_number1 + random_city_number2) / 2)
                    {
                        t = tmp_population[city_iter];
                        tmp_population[city_iter] = tmp_population[random_city_number1+random_city_number2-city_iter];
                        tmp_population[random_city_number1+random_city_number2-city_iter] = t;
                        city_iter++;
                    }
                }

            }
            int city_iter = 0;
            while (city_iter <= city_quantity)
            {
                city.population_route[certain_pop][city_iter] = tmp_population[city_iter];
                city_iter++;
            }
        }
        population_iter++;
    }

}

