#include <bits/stdc++.h>
using namespace std;

int ok1,ok2;

int distance(int x,int y)
{
    return x * x + y * y;
}
int main()
{
    string a,b,c,d;
    cin>>a>>b;
    cin>>c>>d;
    if(a.size() > 5 || b.size() > 5) ok1 = 1;
    if(c.size() > 5 || d.size() > 5) ok2 = 1;
    if(!ok1 && !ok2)
    {
        int x1 = stoi(a);
        int y1 = stoi(b);
        int x2 = stoi(c);
        int y2 = stoi(d);
        if(distance(x1,y1) > (1000 * 1000)) ok1 = 1;
        if(distance(x2,y2) > (1000 * 1000)) ok2 = 1;
        if(!ok1 && !ok2)
        {
            if(distance(x1,y1) < distance(x2,y2))
            {
                puts("Alice");
            }
            else if(distance(x1,y1) > distance(x2,y2))
            {
                puts("Bob");
            }
            else
            {
                puts("Draw");
            }
            return 0;
        }
    }
    if(ok1 && ok2) puts("Draw");
    else if(ok1 && !ok2) puts("Bob");
    else if(!ok1 && ok2) puts("Alice");
    return 0;
}
