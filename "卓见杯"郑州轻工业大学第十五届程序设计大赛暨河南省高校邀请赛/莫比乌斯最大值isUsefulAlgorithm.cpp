#include <bits/stdc++.h>
using namespace std;

typedef long long LL;
const int N = 1e5+10;
int a[N], b[N];
int n;

struct Node
{
    int a,b;
}q[N];
int main()
{
    scanf("%d",&n);
    for(int i = 1;i <= n;i++) scanf("%d",&a[i]);;
    for(int i = 1;i <= n;i++) scanf("%d",&b[i]);;
    sort(a + 1,a + n + 1);
    sort(b + 1,b + n + 1);
    for(int i = 1;i <= n;i++)
    {
        for(int j = 1;j * j <= a[i];j++)
        {
            if(a[i] % j == 0)
            {
                q[j].a = max(q[j].a, a[i]);
                q[a[i] / j].a = max(q[a[i] / j].a, a[i]); 
            }
        }
    }
    for(int i = 1;i <= n;i++)
    {
        for(int j = 1;j * j <= b[i];j++)
        {
            if(b[i] % j == 0)
            {
                q[j].b = max(q[j].b, b[i]);
                q[b[i] / j].b = max(q[b[i] / j].b, b[i]); 
            }
        }
    }
    LL res = 0;
    for(int i = 1;i <= a[n] && b[n];i++)
    {
        res = max(res, (LL)q[i].a * q[i].b * i);
    }
    cout<<res<<endl;
    return 0;
}
