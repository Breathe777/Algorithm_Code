#include<bits/stdc++.h>
using namespace std;

typedef long long LL;
const int N = 5e5 + 10;
int sum[N];
int n,m,k;
int main()
{
	string s;
	char c1,c2;
    cin>>k>>s;
    getchar();
    cin>>c1>>c2;
    n = s.size();
    for(int i = n - 1;i >= 0;i--)
    {
        if(s[i] == c2) sum[i] = 1;
        sum[i] += sum[i+1];
    }
    LL res = 0;
    for(int i = 0;i < n - k + 1;i++)
    {
    	if(s[i] == c1)
    	{
    		int j = i + k - 1;
    		res += sum[j];
		}
	}
	printf("%lld\n",res);
    return 0;
}
