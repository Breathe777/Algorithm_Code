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
