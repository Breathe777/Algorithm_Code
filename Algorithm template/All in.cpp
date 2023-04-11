//快速排序
#include <iostream>
using namespace std;

const int N = 1e5+10;
int a[N];

void quicksort(int left, int right)
{
   if(left >= right) return;
   int x = a[(left + right) >> 1];
   int i = left, j = right - 1;
   while(i <= j)
   {
       while(a[i] < x) i++;
       while(a[j] > x) j--;
       if(i <= j)
       {
           swap(a[i],a[j]);
           i++, j--;
       }
   }
   /* if(k <= j) */ quicksort(left, j + 1);
   /* if(k >= i) */quicksort(i,right);
   /* return a[k]; */
}
int main()
{
   int n;
   scanf("%d",&n);
   for(int i = 0;i < n;i++) scanf("%d",&a[i]);
   quicksort(0,n);
   for(int i = 0;i < n;i++) printf("%d ",a[i]);
   return 0;
} 



//归并排序
#include <iostream>
using namespace std;

const int N = 100010;

int q[N], tmp[N];
int n;

void merge_sort(int left, int right)
{
    if(left >= right) return;
    int mid = (left + right) >> 1;
    merge_sort(left, mid);
    merge_sort(mid + 1,right);
    int k = 0, i = left, j = mid + 1;
    while(i <= mid && j <= right)
    {
        if(q[i] <= q[j]) tmp[k++] = q[i++];
        else tmp[k++] = q[j++];
    }
    while(i <= mid) tmp[k++] = q[i++];
    while(j <= right) tmp[k++] = q[j++];
    for(int i = left, j = 0;i <= right;i++, j++) q[i] = tmp[j];
}
int main()
{
    scanf("%d",&n);
    for(int i = 0;i < n;i++) scanf("%d",&q[i]);
    merge_sort(0,n - 1);
    for(int i = 0;i < n;i++) printf("%d ",q[i]);
    return 0;
}



//堆排序
#include <iostream>
using namespace std;

const int N = 1e5+10;
int h[N],sz;
int n,m;
void down(int u){
   int t = u;
   if(2*u <= sz && h[2*u] < h[t]) t = 2*u;
   if(2*u+1 <= sz && h[2*u+1]<h[t]) t = 2*u+1;
   if(t != u){
       swap(h[t],h[u]);
       down(t);
   }
}
int main()
{
   scanf("%d%d",&n,&m);
   for(int i = 1;i <= n;i++) scanf("%d",&h[i]);
   sz = n;
   for(int i = n/2;i >= 1;i--) down(i);
   
   while(m--){
       printf("%d ",h[1]);
       h[1] = h[sz];
       sz--;
       down(1);
   }
   return 0;
} 



//附带映射关系的手写堆
#include <iostream>
#include <cstring>
using namespace std;

const int N = 100010;
int h[N],ph[N],hp[N],sz;
int n,m;

void swap_heap(int a, int b)
{
    swap(ph[hp[a]], ph[hp[b]]);
    swap(hp[a], hp[b]);
    swap(h[a], h[b]);
}
void down(int u)
{
    int t = u;
    if(2 * u <= sz && h[2 * u] < h[t]) t = 2 * u;
    if(2 * u + 1 <= sz && h[2 * u + 1] < h[t]) t = 2 * u + 1;
    if(t != u)
    {
        swap_heap(t,u);
        down(t);
    }
}
void up(int u)
{
    while(u / 2 && h[u / 2] > h[u])
    {
        swap_heap(u / 2, u);
        u /= 2;
    }
}
int main()
{
    scanf("%d",&n);
    while(n--)
    {
        char op[5];
        scanf("%s",op);
        if(!strcmp(op,"I"))
        {
            int x;
            scanf("%d",&x);
            sz++;
            m++;
            h[sz] = x;
            ph[m] = sz, hp[sz] = m;
            up(sz);
        }
        else if(!strcmp(op,"PM")) printf("%d\n",h[1]);
        else if(!strcmp(op,"DM")) 
        {
            swap_heap(1,sz);
            sz--;
            down(1);
        }
        else if(!strcmp(op,"D"))
        {
            int k;
            scanf("%d",&k);
            k = ph[k];
            swap_heap(k,sz);
            sz--;
            down(k), up(k);
        }
        else
        {
            int k,x;
            scanf("%d%d",&k,&x);
            k = ph[k];
            h[k] = x;
            down(k), up(k);
        }
    }
    return 0;
}



//区间合并
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

typedef pair<int,int> PII;
int n;
void merge(vector<PII>& segs)
{
    vector<PII> ans;
    sort(segs.begin(), segs.end());
    int st = -2e9, ed = -2e9;
    for(PII seg : segs)
    {
        if(ed < seg.first) 
        {
            if(st != -2e9) ans.push_back({st, ed});
            st = seg.first, ed = seg.second;
        }
        else ed = max(ed, seg.second);
    }
    if(st != -2e9) ans.push_back({st, ed});
    
    segs = ans;
}
int main()
{
    cin>>n;
    vector<PII> segs;
    for(int i = 0;i < n;i++)
    {
        int x,y;
        cin>>x>>y;
        segs.push_back({x, y});
    }
    merge(segs);
    cout<<segs.size()<<endl;
    return 0;
}



//KMP算法 
#include <iostream>
using namespace std;

const int N = 1e5+10, M = 1e6+10;
char p[N],s[M];
int ne[M];
int n,m;
int main()
{
   scanf("%d%s",&n,p+1);
   scanf("%d%s",&m,s+1);
   for(int i = 2,j = 0;i <= n;i++)
   {
       while(j && p[i] != p[j+1]) j = ne[j];
       if(p[i] == p[j+1]) j++;
       ne[i] = j;
   }
   for(int i = 1, j = 0;i <= m;i++)
   {
       while(j && s[i] != p[j+1]) j = ne[j];
       if(s[i] == p[j+1]) j++;
       if(j == n) printf("%d ",i - n);
   }
   return 0;
}




//字典树 Trie
#include <iostream>
using namespace std;

const int N = 3e6+10;
int son[N][62],cnt[N],idx;
char str[N];
int getcount(char ch){
   if(ch >= 'a' && ch <= 'z') return ch - 'a';
   else if(ch >= 'A' && ch <= 'Z') return ch - 'A' + 26;
   else return ch - '0' + 52;
}
void insert(char str[]){
   int p = 0;
   for(int i = 0;str[i];i++){
       int u = getcount(str[i]);
       if(!son[p][u]) son[p][u] = ++idx;
       p = son[p][u];
       
   }
   cnt[p]++;
}
int query(char str[]){
   int p = 0;
   for(int i = 0;str[i];i++){
       int u = getcount(str[i]);
       if(!son[p][u]) return 0;
       p = son[p][u];
   }
   return cnt[p];
}
int main()
{
   int T;
   scanf("%d",&T);
   while(T--){
       for(int i=0;i<=idx;i++)
           for(int j=0;j<=63;j++)
               son[i][j]=0;
       for(int i=0;i<=idx;i++)
           cnt[i]=0;
       idx = 0;
       int n,m;
       char ch;
       scanf("%d%d",&n,&m);
       for(int i = 0;i < n;i++){
       	scanf("%s",str);
           insert(str);
       }
       for(int i = 0;i < m;i++) {
       	scanf("%s",str);
           printf("%d\n",query(str));
       }
   }
   return 0;
}





//树状数组 (最长上升子序列和问题)
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

typedef long long LL;
const int N = 1e5+10;
int q[N],w[N];
LL tr[N],res;
int n,m;

int lowbit(int x)
{
   return x & -x;
}
void add(int x,LL c)
{
   for(int i = x;i <= m;i += lowbit(i)) tr[i] = max(tr[i],c);
}
LL query(int x)
{
   LL res = 0;
   for(int i = x;i;i -= lowbit(i)) res = max(res,tr[i]);
   return res;
}
int main()
{
   scanf("%d",&n);
   for(int i = 0;i < n;i++) scanf("%d",&w[i]);
   memcpy(q,w,sizeof(w));
   sort(q,q+n);
   m = unique(q,q+n) - q;
   for(int i = 0;i < n;i++)
   {
       int x = lower_bound(q,q+m,w[i]) - q + 1;
       LL sum = query(x-1) + w[i];
       res = max(sum,res);
       add(x,sum);
   }
   printf("%lld\n",res);
   return 0;
} 





线段树 
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 1e6+10;

int n,m;
LL w[N];
struct Node{
	int l,r;
	LL maxv,tag1,tag2;
	bool used;
}tr[N<<2];
void push_up(int u){
	tr[u].maxv = max(tr[u<<1].maxv,tr[u<<1 | 1].maxv);
}
void push_down(int u){
	if(tr[u].used){
		tr[u<<1].tag1 = tr[u].tag1;
		tr[u<<1 | 1].tag1 = tr[u].tag1;
		tr[u<<1].tag2 = tr[u].tag2;
		tr[u<<1 | 1].tag2 = tr[u].tag2;
		tr[u<<1].maxv = tr[u].tag1 + tr[u].tag2;
		tr[u<<1 | 1].maxv = tr[u].tag1 + tr[u].tag2;
		tr[u<<1].used = tr[u<<1 | 1].used = true;
	}
	else {
		tr[u<<1].tag2 += tr[u].tag2;
		tr[u<<1 | 1].tag2 += tr[u].tag2;
		tr[u<<1].maxv += tr[u].tag2;
		tr[u<<1 | 1].maxv += tr[u].tag2;
	}
	tr[u].used = false;
	tr[u].tag1 = tr[u].tag2 = 0;
}
void build(int u,int l,int r){
	if(l == r) tr[u].l = l,tr[u].r = r,tr[u].maxv = w[r];
	else{
		tr[u].l = l,tr[u].r = r,tr[u].maxv = -1e18;
		int mid = (l + r) >> 1;
		build(u << 1,l,mid);
		build(u << 1 | 1,mid + 1,r);
		push_up(u);
	}
}
void modify1(int u,int l,int r,LL v){
	if(tr[u].l >= l && tr[u].r <= r){
		tr[u].tag1 = v;
		tr[u].tag2 = 0;
		tr[u].maxv = v;
		tr[u].used = true;
		return;
	}
	push_down(u);
	int mid = (tr[u].l + tr[u].r) >> 1;
	if(l <= mid) modify1(u<<1,l,r,v);
	if(r > mid) modify1(u << 1 | 1,l,r,v);
	push_up(u);
}
void modify2(int u,int l,int r,LL v){
	if(tr[u].l >= l && tr[u].r <= r){
		tr[u].tag2 += v;
		tr[u].maxv += v;
		return;
	}
	push_down(u);
	int mid = (tr[u].l + tr[u].r) >> 1;
	if(l <= mid) modify2(u<<1,l,r,v);
	if(r > mid) modify2(u << 1 | 1,l,r,v);
	push_up(u);
}
LL query(int u,int l,int r){
   if(tr[u].l >= l && tr[u].r <= r) return tr[u].maxv;
   push_down(u);
   int mid = (tr[u].l + tr[u].r) >> 1;
   LL res = -1e18;
   if(l <= mid) res = max(res,query(u<<1,l,r));
   if(r > mid) res = max(res,query(u<<1 | 1,l,r));
   return res;
}
int main()
{
   scanf("%d%d",&n,&m);
   for(int i = 1;i <= n;i++) scanf("%lld",&w[i]);
   build(1,1,n);
   while(m--){
   	int op,l,r;
   	LL x;
   	scanf("%d%d%d",&op,&l,&r);
   	if(op == 1) {
   		scanf("%lld",&x);
   		modify1(1,l,r,x);
		}
		else if(op == 2) {
   		scanf("%lld",&x);
   		modify2(1,l,r,x);
		}
		else{
			printf("%lld\n",query(1,l,r));
		}
	}
   return 0;
}



//哈希表拉链法解决冲突
#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>

const int N = 100003;
int h[N],e[N],ne[N],idx;
void insert(int x)
{
	int k = (x % N + N) % N;
	e[idx] = x, ne[idx] = h[k], h[k] = idx++;
}
bool find(int x)
{
	int k = (x % N + N) % N;
	for(int i = h[k];i != -1;i = ne[i])
		if(e[i] == x) return true;
	return false;
}
int main()
{
	int n;
	scanf("%d",&n);
	memset(h,-1,sizeof(h));
	while(n--){
		char op[2];
		int x;
		scanf("%s%d",op,&x);
		if(*op == 'I') insert(x);
		else{
			if(find(x)) puts("Yes");
			else puts("No");
		}
	}
	return 0;
}




//哈希表开放寻址法解决冲突
#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>

const int N = 200003;
int h[N], null = 0x3f3f3f3f;
int find(int x)
{
	int k = (x % N + N) % N;
	while(h[k] != null && h[k] != x){
		k++;
		if(k == N) k = 0;
	}
	return k;
}
int main()
{
	int n;
	scanf("%d",&n);
	memset(h,null,sizeof(h));
	while(n--){
		char op[2];
		int x;
		scanf("%s%d",op,&x);
		int k = find(x);
		if(*op == 'I') h[k] = x;
		else{
			if(h[k] == x) puts("Yes");
			else puts("No");
		}
	}
	return 0;
}






//字符串哈希 
#include <iostream>
#include <cstring>
#include <set>
using namespace std;

const int N = 1e5+10;
int h[N],p[N],P = 131;
char str[N];
unsigned long long get(int l,int r){
	return h[r] - h[l - 1] * p[r - l + 1];
}
int main(){
	int n,k;
	scanf("%s%d",str+1,&k);
	n = strlen(str+1);
	p[0] = 1;
	for(int i = 1;i <= n;i++){
		p[i] = p[i-1]*P;
		h[i] = h[i-1]*P + str[i];
	}
	while(k--){
		int l1,r1,l2,r2;
		scanf("%d%d%d%d",&l1,&r1,&l2,&r2);
		if(get(l1,r1) == get(l2,r2)) puts("Yes");
		else puts("No");
	}
	return 0;
}





//AC自动机 
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 1e6,S = 15,M = 1e6+10;
int tr[N*S][26],cnt[N*S];
char str[M];
int q[N*S],ne[N*S];
int n,idx;

void insert(){
   int p = 0;
   for(int i = 0;str[i];i++){
       int t = str[i] - 'a';
       if(!tr[p][t]) tr[p][t] = ++idx;
       p = tr[p][t];
   }
   cnt[p]++;
}
void build(){
   int hh = 0,tt = -1;
   for(int i = 0;i < 26;i++){
   	if(tr[0][i]) q[++tt] = tr[0][i];
	}
   while(hh <= tt){
       int t = q[hh++];
       for(int i = 0;i < 26;i++){
           int c = tr[t][i];
           if(!c) continue;
           int j = ne[t];
           while(j && !tr[j][i]) j = ne[j];
           if(tr[j][i]) j = tr[j][i];
           ne[c] = j;
           q[++tt] = c;
       }
   }
}
int main()
{
   scanf("%d",&n);
   for(int i = 0;i < n;i++){
       scanf("%s",str);
       insert();
   }
   build();
   scanf("%s",str);
   int res = 0;
   for(int i = 0,j = 0;str[i];i++){
       int t = str[i] - 'a';
       while(j && !tr[j][t]) j = ne[j];
       if(tr[j][t]) j = tr[j][t];
       
       int p = j;
       while(p){
           res += cnt[p];
           cnt[p] = 0;
           p = ne[p];
       }
   }
   printf("%d\n",res);
   return 0;
}



#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 1e6,S = 15,M = 1e6+10;
int tr[N*S][26],cnt[N*S];
char str[M];
int q[N*S],ne[N*S];
int n,idx;

void insert(){
   int p = 0;
   for(int i = 0;str[i];i++){
       int t = str[i] - 'a';
       if(!tr[p][t]) tr[p][t] = ++idx;
       p = tr[p][t];
   }
   cnt[p]++;
}
void build(){
   int hh = 0,tt = -1;
   for(int i = 0;i < 26;i++){
   	if(tr[0][i]) q[++tt] = tr[0][i];
	}
   while(hh <= tt){
       int t = q[hh++];
       for(int i = 0;i < 26;i++){
           int p = tr[t][i];
           if(!p) tr[t][i] = tr[ne[t]][i];
           else{
           	ne[p] = tr[ne[t]][i];
           	q[++tt] = p;
			}
       }
   }
}
int main()
{
   scanf("%d",&n);
   for(int i = 0;i < n;i++){
       scanf("%s",str);
       insert();
   }
   build();
   scanf("%s",str);
   int res = 0;
   for(int i = 0,j = 0;str[i];i++){
       int t = str[i] - 'a';
       j = tr[j][t];
       
       int p = j;
       while(p){
           res += cnt[p];
           cnt[p] = 0;
           p = ne[p];
       }
   }
   printf("%d\n",res);
   return 0;
}





//manacher算法 
#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
using namespace std;

const int N = 2e7;
char a[N],b[N<<1];
int n,p[N<<1];

void init(){
   int k = 0;
   b[k++] = '$',b[k++] = '#';
   for(int i = 0;i < n;i++){
       b[k++] = a[i];
       b[k++] = '#';
   }
   b[k++] = '^';
   n = k;
}
void manacher(){
   int C,R = 0;
   for(int i = 1;i < n;i++){
       if(i < R) p[i] = min(p[(C<<1) - i],R - i);
       else p[i] = 1;
       while(b[i - p[i]] == b[i + p[i]]) p[i]++;
       if(i + p[i] > R){
           R = i + p[i];
           C = i;
       }
   }
}
int main()
{
   scanf("%s",a);
   n = strlen(a);
   init();
   manacher();
   int ans = 1;
   for(int i = 0;i < n;i++) ans = max(ans,p[i]);
   printf("%d\n",ans - 1);
   return 0;
}





//拓扑排序
#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
using namespace std;

const int N = 100010;
int n,m;
int h[N],ne[N],e[N],d[N],q[N],idx;
void add(int a,int b){
	e[idx] = b, ne[idx] = h[a], h[a] = idx++;
}
bool topsort(){
	int start = 0,end = -1;
	for(int i = 1;i <= n;i++)
		if(!d[i]) q[++end] = i;
	while(start <= end){
		int t = q[start++];
		for(int i = h[t];i != -1;i = ne[i]){
			int j = e[i];
			d[j]--;
			if(d[j] == 0) q[++end] = j;
		}
	}
	return end == n - 1;
}
int main()
{
	scanf("%d%d",&n,&m);
	memset(h,-1,sizeof(h));
	for(int i = 0;i < m;i++){
		int a,b;
		scanf("%d%d",&a,&b);
		add(a,b);
		d[b]++; 
	}
	if(topsort()){
		for(int i = 0;i <= n-1;i++) printf("%d ",q[i]);
	} 
	else puts("不存在拓扑排序"); 
}





//朴素Dijkstra(稠密图)
#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
using namespace std;

const int N = 510;
int g[N][N],dist[N];
int n,m;
bool st[N];

int Dijkstra(){
	memset(dist,0x3f,sizeof(dist));
	dist[1] = 0;
	for(int i = 0;i < n - 1;i++){
		int t = -1;
		for(int j = 1;j <= n;j++)
			if(!st[j] && (t == -1 || dist[t] > dist[j])) t = j;
		st[t] = true;
		for(int j = 1;j <= n;j++)
		    dist[j] = min(dist[j],dist[t] + g[t][j]);
	}
	if(dist[n] == 0x3f3f3f3f) return -1;
	else return dist[n];
}
int main()
{
	scanf("%d%d",&n,&m);
	memset(g,0x3f,sizeof(g));
	for(int i = 0;i < m;i++){
		int a,b,c;
		scanf("%d%d%d",&a,&b,&c);
		g[a][b] = min(g[a][b],c);//选最小重边 
	}
	int t = Dijkstra();
	printf("%d\n",t);
}






//堆优化Dijkstra(稀疏图) 
#include <iostream>
#include <cstring>
#include <queue>
using namespace std;

typedef pair<int, int> PII;
const int N = 150010;
int h[N], ne[N], e[N], w[N], idx;
int dist[N];
int n, m;
bool st[N];
void add(int a, int b, int c)
{
   e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}
int Dijkstra()
{
   memset(dist,0x3f,sizeof(dist));
   dist[1] = 0;
   priority_queue<PII, vector<PII>, greater<PII>>heap;
   heap.push({0, 1});
   while(!heap.empty())
   {
       PII t = heap.top();
       heap.pop();
       int ver = t.second, distance = t.first;
       if(st[ver]) continue;
       st[ver] = true;
       for(int i = h[ver];i != -1;i = ne[i])
       {
           int j = e[i];
           if(dist[j] > distance + w[i])
           {
               dist[j] = distance + w[i];
               heap.push({dist[j], j});
           }
       }
   }
   if(dist[n] == 0x3f3f3f3f) return -1;
   else return dist[n];
}
int main()
{
   scanf("%d%d",&n, &m);
   memset(h, -1 , sizeof(h));
   for(int i = 0;i < m;i++)
   {
       int a, b, c;
       scanf("%d%d%d",&a, &b, &c);
       add(a,b,c);
   }
   int t = Dijkstra();
   printf("%d\n",t);
   return 0;
}






//bellman-Ford(负权边)
#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <queue> 
using namespace std;

typedef pair<int,int> PII;
const int N = 510,M = 100010;
int dist[N],backup[N];
int n,m,k;

struct Edge
{
	int a,b,w;
}edges[M]; 
int bellman_Ford()
{
	memset(dist,0x3f,sizeof(dist));
	dist[1] = 0;
	for(int i = 0;i < k;i++){
		memcpy(backup,dist,sizeof(dist));
		for(int j = 0;j < m;j++){
			int a = edges[j].a,b = edges[j].b, w = edges[j].w;
			if(dist[b] > backup[a] + w)
			    dist[b] = backup[a] + w;
		}
	}
	if(dist[n] > 0x3f3f3f3f/2) return 0x3f3f3f3f;
	else return dist[n];
}
int main()
{
	scanf("%d%d%d",&n,&m,&k);
	for(int i = 0;i < m;i++){
		int a,b,w;
		scanf("%d%d%d",&a,&b,&w);
		edges[i].a = a,edges[i].b = b,edges[i].w = w;
	}
	int t = bellman_Ford();
	if(t != 0x3f3f3f3f) printf("%d\n",t);
	else puts("impossible");
}





//SPFA
#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <queue> 
using namespace std;

const int N = 100010;
int h[N],ne[N],e[N],w[N],dist[N],idx;
int n,m;
bool st[N];

void add(int a,int b,int c){
	e[idx] = b, w[idx] = c,  ne[idx] = h[a], h[a] = idx++;
}
int SPFA(){
	memset(dist,0x3f,sizeof(dist));
	dist[1] = 0;
	queue<int>q;
	q.push(1);
	st[1] = true; 
	while(!q.empty()){
		int t = q.front();
		q.pop(),st[t] = false;
		for(int i = h[t];i != -1;i = ne[i]){
			int j = e[i]; 
			if(dist[j] > dist[t] + w[i]){
				dist[j] = dist[t] + w[i];
				if(!st[j]){
					q.push(j);
					st[j] = true;
				}
			}
		}
	}
	if(dist[n] == 0x3f3f3f3f) return 0x3f3f3f3f;
	else return dist[n];
}
int main()
{
	scanf("%d%d",&n,&m);
	memset(h,-1,sizeof(h));
	for(int i = 0;i < m;i++){
		int a,b,c;
		scanf("%d%d%d",&a,&b,&c);
		add(a,b,c);
	}
	int t = SPFA();
	if(t == 0x3f3f3f3f) puts("impossible");
	else printf("%d\n",t);
}






//Floyd算法
#include <iostream>
using namespace std;

const int N = 210, INF = 0x3f3f3f3f;
int g[N][N];
int n,m,k;
void Floyd()
{
   for(int k = 1;k <= n;k++)
   {
       for(int i = 1;i <= n;i++)
       {
           for(int j = 1;j <= n;j++)
           {
               if(g[i][j] > g[i][k] + g[k][j])
                   g[i][j] = g[i][k] + g[k][j];
           }
       }
   }
}
int main()
{
   scanf("%d%d%d",&n,&m,&k);
   for(int i = 1;i <= n;i++)
   {
       for(int j = 1;j <= n;j++)
       {
           if(i == j) g[i][j] = 0;
           else g[i][j] = INF;
       }
   }
   for(int i = 1;i <= m;i++)
   {
       int a,b,c;
       scanf("%d%d%d",&a,&b,&c);
       g[a][b] = min(g[a][b],c);
   }
   Floyd();
   while(k--)
   {
       int x,y;
       scanf("%d%d",&x,&y);
       if(g[x][y] > INF / 2) puts("impossible");
       else printf("%d\n",g[x][y]);
   }
   return 0;
} 







//朴素Prim
#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
using namespace std;

const int N = 510,INF = 0x3f3f3f3f;
int g[N][N],dist[N];
int n,m;
bool st[N];

int Prim()
{
	memset(dist,0x3f,sizeof(dist));
	int res = 0;
	for(int i = 0;i < n;i++){
		int t = -1;
		for(int j = 1;j <= n;j++)
			if(!st[j] && (t == -1 || dist[t] > dist[j])) t = j;
		if(i && dist[t] == INF) return INF;
		if(i) res += dist[t];
		for(int j = 1;j <= n;j++) dist[j] = min(dist[j],g[t][j]);
		st[t] = true;
	} 
	return res;
}
int main()
{
	scanf("%d%d",&n,&m);
	memset(g,0x3f,sizeof(g));
	for(int i = 0;i < m;i++){
		int a,b,c;
		scanf("%d%d%d",&a,&b,&c);
		g[a][b] = g[b][a] = c;
	}
	int t = Prim();
	if(t == INF) printf("impossible");
	else printf("%d\n",t);
 return 0;
}



//Kruskal算法
#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
using namespace std;

const int N = 200010;
int p[N];
int n,m;

struct Edge
{
	int a,b,w;
	bool operator < (const Edge W){
		return w < W.w;
	}
}edges[N];
int find(int u)
{
	if(p[u] != u) p[u] = find(p[u]);
	return p[u];
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i = 0;i < m;i++){
		int a,b,w;
		scanf("%d%d%d",&a,&b,&w);
		edges[i] = {a, b, w};
	}
	sort(edges,edges + m);
	for(int i = 1;i <= n;i++) p[i] = i;
	int res = 0,cnt = 0;
	for(int i = 0;i < m;i++){
		int a = edges[i].a, b = edges[i].b, w = edges[i].w;
		a = find(a), b = find(b);
		if(a != b){
			p[a] = b;
			res += w;
			cnt++;
		}
	}
	if(cnt < n-1) printf("impossible");
	else printf("%d\n",res);
	return 0;
}

 
 
 
 
//染色法判断二分图
#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
using namespace std;

const int N = 100010, M = 200010;
int h[N],ne[M],e[M],idx,color[N];
int n,m;
void add(int a,int b)
{
	e[idx] = b, ne[idx] = h[a], h[a] = idx++;
}
bool dfs(int u,int c)
{
	color[u] = c;
	for(int i = h[u];i != -1;i = ne[i]){
		int j = e[i];
		if(!color[j]){
			if(!dfs(j,3 - c)) return false;
		}
		else if(color[u] == color[j]) return false;
	}
	return true;
}
int main()
{
	scanf("%d%d",&n,&m);
	memset(h,-1,sizeof(h));
	for(int i = 0;i < m;i++){
		int a,b;
		scanf("%d%d",&a,&b);
		add(a,b), add(b,a);
	}
	bool flag = true;
	for(int i = 1;i <= n;i++){
		if(!color[i]){
			if(!dfs(i,1)){
				flag = false;
				break;
			}
		}
	}
	if(flag) puts("Yes");
	else puts("No");
}



 

//匈牙利算法寻找二分图最大匹配
#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
using namespace std;

const int N = 510, M = 100010;
int n1,n2,m;
int h[N],ne[M],e[M],idx;
int match[N];
bool st[N];

void add(int a,int b){
	e[idx] = b, ne[idx] = h[a], h[a] = idx++;
}
int find(int x){
	for(int i = h[x];i != -1;i = ne[i]){
		int j = e[i];
		if(!st[j]){
			st[j] = true;
			if(!match[j] || find(match[j])){
				match[j] = x;
				return true;
			}
		}
	}
	return false;
}
int main()
{
	scanf("%d%d%d",&n1,&n2,&m);
	memset(h,-1,sizeof(h));
	while(m--){
		int a,b;
		scanf("%d%d",&a,&b);
		add(a,b);
	}
	int res = 0;
	for(int i = 1;i <= n1;i++){
		memset(st,false,sizeof(st));
		if(find(i)) res++;
	}
	printf("%d\n",res);
}




//试除法判定质数 
#include <iostream>
using namespace std;

const int N = 110;
bool is_prime(int n){
   if(n < 2) return false;
   for(int i = 2;i <= n/i;i++){
       if(n % i == 0) return false;
   }
   return true;
}
int main()
{
   int n,x;
   scanf("%d",&n);
   for(int i = 0;i < n;i++){
       scanf("%d",&x);
       if(is_prime(x)) puts("Yes");
       else puts("No");
   }
   return 0;
}



//分解质因数
#include <iostream>
using namespace std;

void divide(int n)
{
   for(int i = 2;i <= n / i;i++){
       if(n % i == 0)
       {
           int s = 0;
           while(n % i == 0){
               s++;
               n /= i;
           }
           printf("%d %d\n",i,s);
       }
   }
   if(n > 1) printf("%d %d\n",n,1);
   puts("");
}
int main()
{
   int n;
   scanf("%d",&n);
   for(int i = 0;i < n;i++){
       int x;
       scanf("%d",&x);
       divide(x);
   }
   return 0;
} 



//质数筛  埃氏筛 
#include <iostream>
using namespace std;

const int N =  1e6+10;
int primes[N],cnt;
bool st[N];
void get_prime(int n)
{
	for(int i = 2;i <= n;i++){
		if(!st[i]){
			primes[cnt++] = i;
			for(int j = i + i;j <= n;j += i){
				st[j] = true;
			}
		}
	}
	for(int i = 0;i < cnt;i++){
		printf("%d ",primes[i]);
	}
}
int main()
{
	int n;
	scanf("%d",&n);
	get_prime(n);
	return 0;
}


//欧拉筛,线性筛 
#include <iostream>
using namespace std;

const int N =  1e6+10;
int primes[N],cnt;
bool st[N];
void get_prime(int n)
{
	for(int i = 2;i <= n;i++){
		if(!st[i]){
			primes[cnt++] = i;
		}
		for(int j = 0;primes[j] <= n / i;j++){
			st[primes[j] * i] = true;
			if(i % primes[j] == 0) break;
		} 
	}
	for(int i = 0;i < cnt;i++) printf("%d ",primes[i]);
}
int main()
{
	int n;
	scanf("%d",&n);
	get_prime(n);
	return 0;
}




//试除法求约数
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

const int N = 110;
vector<int> get_divisors(int n){
   vector<int>ve;
   for(int i = 1;i <= n/i;i++){
       if(n % i == 0){
           ve.push_back(i);
           if(i != n / i) ve.push_back(n/i);
       }
   }
   sort(ve.begin(),ve.end());
   return ve;
}
int main()
{
   int n;
   scanf("%d",&n);
   for(int i = 0;i < n;i++){
       int a;
       scanf("%d",&a);
       auto res = get_divisors(a);
       for(auto x : res) printf("%d ",x);
       puts("");
   }
   return 0;
} 


//约数个数 
#include <iostream>
#include <unordered_map>
using namespace std;

typedef long long LL;
const int mod = 1e9+7;
int n;
int main()
{
   unordered_map<int,int>primes;
   scanf("%d",&n);
   while(n--){
       int x;
       scanf("%d",&x);
       for(int i = 2;i <= x/i;i++){
           while(x % i == 0){
               x /= i;
               primes[i]++;
           }
       }
       if(x > 1) primes[x]++;
   }
   LL res = 1;
   for(auto prime : primes) res = res*(prime.second + 1) % mod;
   printf("%lld",res);
   return 0;
}


//约数之和
#include <iostream>
#include <unordered_map>
using namespace std;

typedef long long LL;
const int mod = 1e9+7;
int n;
int main()
{
   unordered_map<int,int>primes;
   scanf("%d",&n);
   while(n--){
       int x;
       scanf("%d",&x);
       for(int i = 2;i <= x/i;i++){
           while(x % i == 0){
               x /= i;
               primes[i]++;
           }
       }
       if(x > 1) primes[x]++;
   }
   LL res = 1;
   for(auto prime : primes){
       LL t = 1;
       int p = prime.first, a = prime.second;
       while(a--) t = (t * p + 1) % mod;
       res = res * t % mod;
   }
   printf("%lld",res);
   return 0;
} 
 
 
 
 
//欧几里得算法
#include <iostream>
using namespace std;

int my_gcd(int a,int b){
   return b ? my_gcd(b,a % b) : a;
}
int main()
{
   int n;
   scanf("%d",&n);
   while(n--){
       int a,b;
       scanf("%d%d",&a,&b);
       printf("%d\n",my_gcd(a,b));
   }
   return 0;
} 




//快速幂
#include <iostream>
using namespace std;

long long quick_pow(long long a,int b,int p){
   long long res = 1;
   while(b){
       if(b & 1) res = res * a % p;
       a = a * a % p;
       b >>= 1;
   }
   return res;
}
int main()
{
   int n;
   long long res;
   scanf("%d",&n);
   while(n--){
       int a,b,p;
       scanf("%d%d%d",&a,&b,&p);
       res = quick_pow(a,b,p);
       printf("%lld\n",res);
   }
   return 0;
} 






//欧拉函数   (求与N互质个数)
#include <iostream>
using namespace std;

int main()
{
   int n;
   scanf("%d",&n);
   while(n--){
       int a;
       scanf("%d",&a);
       int res = a;
       for(int i = 2; i <= a/i;i++){
           if(a % i == 0)
           {
               res = res / i * (i - 1);
               while(a % i == 0) a /= i;
           }
       }
       if(a > 1) res = res / a * (a - 1);
       printf("%d\n",res);
   }
   return 0;
}
 

  
 
 
 
//筛法求欧拉函数   若a与n互质,则a的f(n)次方mod n = 1,(f为欧拉函数)
#include <iostream>
using namespace std;

typedef long long LL;
const int N = 1e6+10;
int primes[N],cnt;
int phi[N];
bool st[N];
LL get_eulers(int n)
{
   phi[1] = 1;
   for(int i = 2;i <= n;i++){
       if(!st[i]){
           primes[cnt++] = i;
           phi[i] = i - 1;
       }
       for(int j = 0;primes[j] <= n / i;j++){
           st[primes[j] * i] = true;
           if(i % primes[j] == 0){
               phi[primes[j] * i] = phi[i] * primes[j];
               break;
           }
           phi[primes[j] * i] = phi[i] * (primes[j] - 1);
       }
   }
   LL res = 0;
   for(int i = 1;i <= n;i++) res += phi[i];
   return res;
}
int main()
{
   int n;
   scanf("%d",&n);
   printf("%lld",get_eulers(n));
}









//拓展欧几里得  (ax + by = d, d能整除gcd(a,b)有解,否则无解)
//求的是x0,y0特解，通解   x = x0 - b/d*k, y = y0 + b/d*k   (k∈Z)

拓展欧几里得求逆元
a,p互素, 因为 (a * x) mod p = 1, (k * p + 1) mod p = 1, 所以 a * x = k * p + 1 ---> a * x + k * p = 1(拓展欧几里得)
(p为非质数用拓展欧几里得求逆元,p为质数用费马小定理求逆元)   求得任意一组满足等式的 x 和 k ,其中的x就是a的逆元

#include <iostream>
using namespace std;

int exgcd(int a,int b,int &x,int &y){
   if(!b){
       x = 1,y = 0;
       return a;
   }
   int d = exgcd(b, a % b, y, x);
   y = y - a / b * x;
   return d;
}
int main()
{
   int n;
   scanf("%d",&n);
   while(n--){
       int a,b,x,y;
       scanf("%d%d",&a,&b);
       exgcd(a,b,x,y);
       printf("%d %d\n",x,y);
   }
   return 0;
} 






//高斯消元 
#include <iostream>
#include <cmath>
using namespace std;

const int N = 110;
const double eps = 1e-6;
double a[N][N];
int n;

int gauss()
{
   int c, r;
   for (c = 0, r = 0; c < n; c ++ )
   {
       int t = r;
       for (int i = r; i < n; i ++ )   // 找到绝对值最大的行
           if (fabs(a[i][c]) > fabs(a[t][c]))
               t = i;

       if (fabs(a[t][c]) < eps) continue;

       for (int i = c; i <= n; i ++ ) swap(a[t][i], a[r][i]);      // 将绝对值最大的行换到最顶端
       for (int i = n; i >= c; i -- ) a[r][i] /= a[r][c];      // 将当前上的首位变成1
       for (int i = r + 1; i < n; i ++ )       // 用当前行将下面所有的列消成0
           if (fabs(a[i][c]) > eps)
               for (int j = n; j >= c; j -- )
                   a[i][j] -= a[r][j] * a[i][c];

       r ++ ;
   }

   if (r < n)
   {
       for (int i = r; i < n; i ++ )
           if (fabs(a[i][n]) > eps)
               return 2; // 无解
       return 1; // 有无穷多组解
   }

   for (int i = n - 1; i >= 0; i -- )
       for (int j = i + 1; j < n; j ++ )
           a[i][n] -= a[i][j] * a[j][n];

   return 0; // 有唯一解
}
int main()
{
   scanf("%d",&n);
   for(int i = 0; i < n;i++)
       for(int j = 0;j < n + 1;j++)
           scanf("%lf",&a[i][j]);
   int t = gauss();
   if(t == 0){
       for(int i = 0;i < n;i++) printf("%.2lf\n",a[i][n]);
   }
   else if(t == 1) puts("Infinite group solutions");
   else puts("No solution");
   return 0;
}






//求组合数Ⅰ
#include <iostream>
using namespace std;

const int N = 2010, mod = 1e9+7;
int c[N][N];
void init()
{
   for(int i = 0;i < N;i++){
       for(int j = 0;j <= i;j++){
           if(!j) c[i][j] = 1;
           else c[i][j] = (c[i-1][j] + c[i-1][j-1]) % mod;
       }
   }
}
int main()
{
   int n;
   scanf("%d",&n);
   init();
   while(n--){
       int a,b;
       scanf("%d%d",&a,&b);
       printf("%d\n",c[a][b]);
   }
   return 0;
} 



//组合数Ⅱ
#include <iostream>
using namespace std;

typedef long long LL;
const int N = 100010, mod = 1e9+7;

int fact[N], infact[N];

int qmi(int a,int b,int p){
   int res = 1;
   while(b){
       if(b & 1) res = (LL)res * a % p;
       a = (LL)a * a % p;
       b = b>>1;
   }
   return res;
}
int main()
{
   fact[0] = 1;
   infact[0] = 1;
   for(int i = 1;i < N;i++){
       fact[i] = (LL)fact[i-1] * i % mod;
       infact[i] = (LL)infact[i-1] * qmi(i,mod-2,mod) % mod;
   }
   int n;
   scanf("%d",&n);
   while(n--){
       int a,b;
       scanf("%d%d",&a,&b);
       printf("%d\n",(LL)fact[a] * infact[b] % mod * infact[a-b] % mod);
   }
   return 0;
}
 


//组合数Ⅲ
#include <iostream>
using namespace std;

typedef long long LL;
int p;

int qmi(int a,int b){
   int res = 1;
   while(b)
   {
       if(b & 1) res = (LL)res * a % p;
       a = (LL)a * a % p;
       b = b>>1;
   }
   return res;
}
int C(int a,int b){
   int res = 1;
   for(int i = 1, j = a;i <= b;i++,j--){
       res = (LL)res * j % p;
       res = (LL)res * qmi(i,p-2) % p;
   }
   return res;
}
int lucas(LL a,LL b){
   if(a < p && b < p) return C(a,b);
   else return (LL)C(a%p,b%p)*lucas(a/p,b/p)%p;
}
int main()
{
   int n;
   scanf("%d",&n);
   while(n--){
       LL a,b;
       scanf("%d%d%d",&a,&b,&p);
       printf("%d\n",lucas(a,b));
   }
   return 0;
}




//组合数Ⅳ
#include <iostream>
#include <vector>
using namespace std;
const int N = 5100;
int sum[N],primes[N],cnt;
bool st[N];

void get_primes(int n)
{
   for(int i = 2;i <= n;i++){
       if(!st[i]) primes[cnt++] = i;
       for(int j = 0;primes[j] * i <= n;j++)
       {
           st[primes[j] * i] = true;
           if(i % primes[j] == 0) break;
       }
   }
}
int get(int n,int p)
{
   int res = 0;
   while(n)
   {
       res += n / p;
       n /= p;
   }
   return res;
}
vector<int> mul(vector<int>a, int b)
{
   vector<int>c;
   int t = 0;
   for(int i = 0;i < a.size();i++){
       t += a[i] * b;
       c.push_back(t % 10);
       t /= 10;
   }
   while(t){
       c.push_back(t % 10);
       t /= 10;
   }
   return c;
}
int main()
{
   int a,b;
   scanf("%d%d",&a,&b);
   get_primes(a);
   
   for(int i = 0;i < cnt;i++){
       int p = primes[i];
       sum[i] = get(a,p) - get(b,p) - get(a - b,p);
   }
   vector<int>res;
   res.push_back(1);
   for(int i = 0; i < cnt;i++)
       for(int j = 0;j < sum[i];j++)
           res = mul(res,primes[i]);
   for(int i = res.size() - 1;i >= 0;i--) printf("%d",res[i]);
   puts("");
   return 0;
} 





//高精度加法
#include <iostream>
#include <vector>
#include <string>
using namespace std;

vector<int> add(vector<int> &A, vector<int> &B)
{
   if(A.size() < B.size()) return add(B,A);
   vector<int> C;
   int t = 0;
   for(int i = 0;i < A.size();i++){
       t += A[i];
       if(i < B.size()) t += B[i];
       C.push_back(t % 10);
       t /= 10;
   }
   while(t){
       C.push_back(t % 10);
       t /= 10;
   }
   return C;
}
int main()
{
   string a,b;
   getline(cin,a);
   getline(cin,b);
   vector<int> A;
   vector<int> B;
   for(int i = a.size() - 1;i >= 0;i--) A.push_back(a[i] - '0');
   for(int i = b.size() - 1;i >= 0;i--) B.push_back(b[i] - '0');
   vector<int> C = add(A,B);
   for(int i = C.size() - 1;i >= 0;i--) printf("%d",C[i]);
   puts("");
   return 0;
}




//高精度减法
#include <iostream>
#include <string>
#include <vector>
using namespace std;

bool cmp(vector<int> &A, vector<int> &B)
{
   if(A.size() != B.size())  return A.size() > B.size();
   for(int i = A.size() - 1;i >= 0;i--)
   {
       if(A[i] != B[i]) return A[i] > B[i];
   }
   return true;
}
vector<int> sub(vector<int>& A, vector<int>& B)
{
   vector<int> C;
   int t = 0;
   for(int i = 0;i < A.size();i++){
       t = A[i] - t;
       if(i < B.size()) t -= B[i];
       C.push_back((t + 10) % 10);
       if(t < 0) t = 1;
       else t = 0;
   }
   while(C.size() > 1 && C.back() == 0) C.pop_back();
   return C;
}
int main()
{
   string a,b;
   getline(cin,a);
   getline(cin,b);
   vector<int> A;
   vector<int> B;
   for(int i = a.size() - 1;i >= 0;i--) A.push_back(a[i] - '0');
   for(int i = b.size() - 1;i >= 0;i--) B.push_back(b[i] - '0');
   vector<int> C;
   if(cmp(A,B)){
       C = sub(A,B);
   }
   else{
       C = sub(B,A);
       printf("-");
   }
   for(int i = C.size() - 1;i >= 0;i--) printf("%d",C[i]);
   puts("");
   return 0;
}



//高精度乘法
#include <iostream>
#include <vector>
#include <string>
using namespace std;

vector<int> mul(vector<int> &A,int b)
{
   vector<int> C;
   int t = 0;
   for(int i = 0;i < A.size();i++){
       t += A[i] * b;
       C.push_back(t % 10);
       t /= 10;
   }
   while(t)
   {
       C.push_back(t % 10);
       t /= 10;
   }
   while(C.size() > 1 && C.back() == 0) C.pop_back();
   return C;
}
int main()
{
   string a;
   getline(cin,a);
   int b;
   scanf("%d",&b);
   vector<int> A;
   for(int i = a.size() - 1;i >= 0;i--) A.push_back(a[i] - '0');
   vector<int> C = mul(A, b);
   for(int i = C.size() - 1;i >= 0;i--) printf("%d",C[i]);
   puts("");
   return 0;
} 



//高精度除法
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
using namespace std;

vector<int> div(vector<int> &A, int b,int &r)
{
   vector<int> C;
   for(int i = A.size() - 1;i >= 0;i--)
   {
       r = r * 10 + A[i];
       C.push_back(r / b);
       r = r % b;
   }
   reverse(C.begin(), C.end());
   while(C.size() > 1 && C.back() == 0) C.pop_back();
   return C;
}
int main()
{
   string a;
   int b;
   getline(cin,a);
   scanf("%d",&b);
   vector<int> A;
   for(int i = a.size() - 1;i >= 0;i--) A.push_back(a[i] - '0');
   int r = 0;
   vector<int> C = div(A, b ,r);
   for(int i = C.size() - 1;i >= 0;i--) printf("%d",C[i]);
   puts("");
   printf("%d\n",r);
   return 0;
}



//倍增求lca 最近公共祖先
#include <iostream>
#include <cstring>
#include <queue>
using namespace std;

const int N = 40010, M = 2 * N;
int h[N], e[M], ne[M], idx;
int depth[N], fa[N][16];
int n,m;

void add(int a,int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx++;
}
void bfs(int root)
{
    memset(depth,0x3f,sizeof depth);
    depth[0] = 0, depth[root] = 1;
    queue<int>q;
    q.push(root);
    while(!q.empty())
    {
        int t = q.front();
        q.pop();
        for(int i = h[t];i != -1;i = ne[i])
        {
            int j = e[i];
            if(depth[j] > depth[t] + 1)
            {
                depth[j] = depth[t] + 1;
                q.push(j);
                fa[j][0] = t;
                for(int k = 1;k <= 15;k++) fa[j][k] = fa[fa[j][k-1]][k-1];
            }
        }
    }
}
int lca(int a,int b)
{
    if(depth[a] < depth[b]) swap(a,b);
    for(int k = 15;k >= 0;k--)
    {
        if(depth[fa[a][k]] >= depth[b])
            a = fa[a][k];
    }
    if(a == b) return b;
    for(int k = 15;k >=0;k--)
    {
        if(fa[a][k] != fa[b][k])
        {
            a = fa[a][k];
            b = fa[b][k];
        }
    }
    return fa[a][0];
}
int main()
{
    scanf("%d",&n);
    memset(h,-1,sizeof h);
    int root;
    for(int i = 0;i < n;i++)
    {
        int a,b;
        scanf("%d%d",&a,&b);
        if(b == -1) root = a;
        else add(a,b), add(b,a);
    }
    bfs(root);
    scanf("%d",&m);
    while(m--)
    {
        int x,y;
        scanf("%d%d",&x,&y);
        int p = lca(x,y);
        if(p == x) puts("1");
        else if(p == y) puts("2");
        else puts("0");
    }
    return 0;
}
