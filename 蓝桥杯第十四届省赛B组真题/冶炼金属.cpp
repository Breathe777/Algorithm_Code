#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

typedef long long LL;
const int N = 1e4 + 10;
int main()
{
	int n;
    scanf("%d",&n);
    int minv = 0,maxv = 1e9;
    for(int i = 0;i < n;i++)
    {
    	int x,y;
    	scanf("%d%d",&x,&y);
    	maxv = min(maxv, x / y);
    	minv = max(minv, x / (y + 1) + 1);
	}
	printf("%d %d\n",minv,maxv);
    return 0;
}
