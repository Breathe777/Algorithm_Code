#include <bits/stdc++.h>
using namespace std;

const int N = 2e5+10;
int a[N], b[N];
string s;
vector<string>ans;
int h[N],e[N],ne[N],idx;
int n;

void add(int a,int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx++;
}
void dfs(int u,int cnt)
{
    if(cnt > 9) return;
    s += b[u] + '0';
    for(int i = h[u];i != -1;i = ne[i])
    {
        int j = e[i];
        dfs(j,cnt + 1);
    }
}
int main()
{
    scanf("%d",&n);
    memset(h,-1,sizeof(h));
    for(int i = 1;i <= n;i++)
    {
        scanf("%d",&a[i]);
        add(i,a[i]);
    }
    int maxb = 0;
    for(int i = 1;i <= n;i++)
    {
        scanf("%d",&b[i]);
        maxb = max(b[i],maxb);
    }
    for(int i = 1;i <= n;i++)
    {
        if(b[i] == maxb)
        {
            s = "";
            dfs(i,1);
            ans.push_back(s);
        }
    }
    sort(ans.begin(),ans.end(),greater<string>());
    cout<<ans[0]<<endl;
    return 0;
}
