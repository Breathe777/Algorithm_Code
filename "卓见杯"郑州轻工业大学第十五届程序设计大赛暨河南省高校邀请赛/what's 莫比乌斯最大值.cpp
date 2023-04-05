#include <bits/stdc++.h>
using namespace std;

typedef long long LL;
const int N = 1e5+10;
unordered_map<string,int>ask;
unordered_map<string,int>answer;
int n,res;


int main()
{
    string s;
    scanf("%d",&n);
    while(n--)
    {
        cin>>s;
        if(s == "what's")
        {
            cin>>s;
            ask[s] = 1;
        }
        else
        {
            string op = "",ans;
            int id = -1;
            for(int i = 0;i < s.size();i++)
            {
                op += s[i];
                if(ask.count(op) && ask[op] == 1 && answer.count(op) == 0)
                {
                    id = i;
                    ans = op;
                }
            }
            if(id != -1)
            {
                answer[ans] = 1;
                res++;
            }
        }
    }
    cout<<res<<endl;
    return 0;
}
