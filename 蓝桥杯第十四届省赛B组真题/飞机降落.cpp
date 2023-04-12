#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

typedef pair<int,int> PII;
typedef long long LL;
const int  N = 20;
int n;
int a[N];
struct Node
{
	int  t,d,l;
}nodes[N];
bool check()
{
	int end = nodes[a[0]].l + nodes[a[0]].t;
	for(int i = 1;i < n;i++)
	{
		int t = nodes[a[i]].t, d = nodes[a[i]].d, l = nodes[a[i]].l;
		if(t + d < end) return false;
		else end = max(end, t) + l;		
	}
	return true;
}
int main()
{
	int T;
	scanf("%d",&T);
	while(T--)
	{
		bool flag = false;
		scanf("%d",&n);
		for(int i = 0;i < n;i++)
		{
			scanf("%d%d%d",&nodes[i].t,&nodes[i].d,&nodes[i].l);
			a[i] = i;
		}
		do
		{
		    if(check()) flag = true;
		}while(next_permutation(a,a+n));
		flag ? puts("YES") : puts("NO");
	}
    return 0;
}
