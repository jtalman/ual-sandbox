/* Inverse of a n by n matrix */
#include<stdio.h>
//#include<conio.h>
#include<math.h>
#include<stdlib.h>
void main(){
 void arg(double *,double *, int *,int ,int );

 double det(double *,int *);
 //int det(int *,int *);
 
 double a[10][10],b[10][10],c[10][10],       d;
   int                               n,i,j,m  ;
   int                                 k      ;
 double tmp                                   ;
 //int a[10][10],b[10][10],c[10][10],n,i,j,m,d;
 
 //clrscr();
 printf("Enter the order of the matrix");
 scanf("%d",&n);
 printf("Enter the matrix");
 for(i=0;i<n;i++){
  for(j=0;j<n;j++){
   scanf("%lf",&a[i][j]);
  }
  //scanf("%d",&a[i][j]);
 }

 if(n==2){
  c[0][0]=a[1][1];
  c[1][1]=a[0][0];
  c[0][1]=-a[0][1];
  c[1][0]=-a[1][0];
  d=a[0][0]*a[1][1]-a[0][1]*a[1][0];
  printf("Determinant is:%e\n",d);
  if(d==0){
   getchar();
   exit(d-'0');
  }

  for(i=0;i<n;i++){
   printf("\n");
   for(j=0;j<n;j++){
    printf(" %f",c[i][j]/(float)d);
   }
  }
 }
 else{
  m=n;
  for(i=0;i<m;i++){
   for(j=0;j<m;j++){
    n=m;
    arg(&a[0][0],&b[0][0],&n,i,j);
    c[j][i]=pow(-1,(i+j))*det(&b[0][0],&n);
   }
  }
  n=m;
  d=det(&a[0][0],&n);
  printf("Determinant is :%e\n",d);
  if(d==0){
   printf("INVERSE DOES NOT EXIST");
   getchar();
   exit(d-'0');
  }
  for(i=0;i<m;i++){
   printf("\n");
   for(j=0;j<m;j++){
    printf(" %f",c[i][j]/(float)d);
   }
  }
 } getchar();

 printf("\n");
 printf("\n");

 for(i=0;i<m;i++){
  for(j=0;j<m;j++){
   tmp=0;
   for(k=0;k<m;k++){
    tmp=tmp+a[i][k]*c[k][j];
   }
   printf(" %f",tmp);
  }
  printf("\n");
 }

 printf("\n");
 printf("\n");

 for(i=0;i<m;i++){
  for(j=0;j<m;j++){
   tmp=0;
   for(k=0;k<m;k++){
    tmp=tmp+c[i][k]*a[k][j];
   }
   printf(" %f",tmp);
  }
  printf("\n");
 }

}

void arg(double *a,double *b,int *n,int x,int y)
{
int k,l,i,j;
for(i=0,k=0;i<*n;i++,k++)
{
for(j=0,l=0;j<*n;j++,l++)
{
if(i==x)
i++;
if(j==y)
j++;
*(b+10*k+l)=*(a+10*i+j);

}
}
*n=*n-1;
}

double det(double *p,int *n)
//int det(int *p,int *n)
{
double d[10][10],      sum=0;
int           i,j,m      ;
m=*n;
if(*n==2)
return(*p**(p+11)-*(p+1)**(p+10));
for(i=0,j=0;j<m;j++)
{
*n=m;
arg(p,&d[0][0],n,i,j);
sum=sum+*(p+10*i+j)*pow(-1,(i+j))*det(&d[0][0],n);
}

return(sum);
}
