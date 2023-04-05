#include <bits/stdc++.h>
using namespace std;

const int N = 2e5+10;
vector<int> v[N];
int ans[N];
priority_queue<int>q;
int n,m,k,res;

int main()
{
    scanf("%d%d%d",&n,&m,&k);
    for(int i = 0;i < m;i++)
    {
        int l,r;
        scanf("%d%d",&l,&r);
        v[l].push_back(r);
    }
    int s = 0;
    for(int i = 1;i <= n;i++)
    {
        int op = 0;
        for(int j = 0;j < v[i].size();j++)
        {
            op++;
            s++;
            ans[v[i][j]]++;
            q.push(v[i][j]);
        }
        while(s > k)
        {
            int t = q.top();
            q.pop();
            op--;
            s--;
            ans[t]--;
        }
        res += op;
        s -= ans[i];
    }
    cout<<res<<endl;
    return 0;
}
