# STAT 433

## Lecture 1

$F$ is a collection of sets (subsets) of $\Omega$.

$P: F\to [0, 1]$

$\sigma$-Field on $\Omega$ is $F\sube \delta^\Omega$  
i) $\empty\in F$  
ii) $A\in F\to A^C\in F$  
iii) $(A_i)_{i\in N}\in F\to\bigcup_{i\in N}A_i\in F$

If $F$ is a $\sigma$-Field on $\Omega$, $(\Omega, F)$ is called a measurable space.

A measure on $(\Omega, F)$ is a function, $\mu: F\to\R_{\geq 0}$ s.t.  
i) $\mu(\empty) = 0$  
ii) $\mu(\bigsqcup_{i\in N} = \sum_{i\in N}\mu | A_i)$  
iii) $(A_i)_{i\in N}$ have $A_i\cap A_j = \empty\forall i, j$

A measurable $\mu$ is finite if $\mu(\Omega)\in\R_{\geq 0}$ on $(\Omega, F)$.

$\sigma$-Finite if $\exists(\Omega_i)_{i\in N}\in F$, $\Omega=\bigcup_{j\in N}\Omega_j$,
$\mu(\Omega_j)<\infty$.

Measure $\mu$ is a Probability measure if $\mu(\Omega) = 1$.

A measure space is a triple $(\Omega, F, \mu)$  
Probability is a triple $(\Omega, F, P)$ where $P$ is a prob measure in $F$.

If $(\Omega_1, F_1)$ and $(\Omega_2, F_2)$ are measurable spaces, then $(\Omega_1\otimes\Omega_2, F_1\otimes F_2)$.

$F_1\otimes F_2$ has all the sets $A\times B$ s.t. $A\in F_1$ and $B\in F_2$.

A random element taking  values in $\chi, F_\chi$ is a function: $X:\Omega\to\chi$
s.t. $X^{-1}(A)\in F\forall A\in F_X$ where $(\Omega, F)$ is a measurable space.

Basically a random variable is a function lol.

Indicator r.v. for set $A\in F$ is $\bold 1_A: \Omega\to\{0, 1\}$.  
$\bold 1_A(w) = \begin{cases}
1 & w\in A \\
0 & o.w\end{cases}$

$\bold 1^{-1}_A(B)=\begin{cases}
\empty & 0,1\notin B \\
A & 1\in B, 0\notin B \\
A^C & 0\in B & 1\notin B \\
\Omega & o.w
\end{cases}$

If $(\Omega, F, \mu)$ is a measure space. Then $\int\cdot du: f\mapsto\int f(w)\mu(dw)$ is
is the unique functional s.t.  
i) $\int\bold 1_a(w)\mu(dw)=\mu(A)$  
ii) $\int(af+g)(w)\mu(dw)=a\int f(w)\mu(dw) + \int g(w)\mu(dw)$

If $(f_n)_{n\in N}$ is sequence of measure functions and $f_n(w)\le f_{n+1}(w)\forall w, n, f_n\geq 0$ then $\int\lim_{n\to\infty}f_n(w)\mu(dw)=\lim_{n\to\infty}\int f_n(w)\mu(dw)$

$EX$ is $\int X(w)P(dw)$.

If $X\geq 0, EX\geq 0$.

$P(X>r)=E(\bold 1_{[x>r]})\le E\bold 1_{[X>r]}(\frac{X}{r})\lt E(\frac{X}{r}) = \frac{EX}{r}$

### Markov's Inequality

If $X$ is a non-neg r.v. Then $P(X>r)\le\frac{EX}{r}$.

### Jensen's Inequality

A function $\phi$ is convex if $\phi(tx_1+(1 -t)x_2)\lt t\phi(x_1) + (1 - t)\phi(x_2)\forall t\in[0,1],x_1,x_2$. Basically, a chord will always lie above its tangent lines if it goes
through a convex function $\phi$.

If $\empty$ convex then $E\phi(X)\geq\phi(EX)$.

Pf.  
$E\phi(X)\geq E[\phi(E(X)) + \partial\phi(EX)(X - EX)] = \phi(EX)$

$\int$ are monotone. $f\geq g \int fdu\geq\int gdu$

 Suppose $f:\Omega_1\otimes\Omega_2\to\R$  
$\int\int f(w_1, w_2)\mu_1(dw_1)\mu_2(dw_2) = \int\int f(w_1, w_2)\mu2(dw_2)\mu1(dw_1)$

Works if $f\geq 0$ or if $\int\int |f|du_1du_2\lt\infty$

Suppose wew ant $EX$ for $X\geq 0$  

$EX=\int_0^\infty P(X>t)dt$

Pf:  
$$
EX=E\Big[\int_0^X 1 dt\Big] \\
= E\Big[\int_0^\infty\bold 1_{t < X}dt\Big] \\
= \int^\infty_0 E[\bold 1_{t<X}] dt \\
= \int_0^\infty P(X>t)dt
$$

Similarly, $EX=\sum_{n\in\N}P(X\geq n)$

* Convergence in probability
* Convergence almost surely aka with probability 1
* Convergence in $L^P: E|X_n - X_\infty|^P\to 0$
* WLLN
* SLLN

If $P_n$ is a sequence of probability measures on $\R$.

$P_n\xrightarrow{D} P_\infty$ if $\forall$ bdd cts $E_nf\to E_\infty f$

$P_n\to P_\infty$ in TV. if $\text{sup}_{A\in F} |P_n(A) - P_\infty(A)|\to 0.$

### Fatou's Lemma

Let $\Omega=[0,1], X_n=\bold 1_{(0, \frac{1}{n}]}\cdot n, P((a,b]) = b-a$

$\forall w X^{(w)}_n\to 0, EX_n = 1, EX_\infty=0$ This is an example of $X_n\to X$ but
$EX_n\nrightarrow EX_\infty$

## Lec 2

### Partial Expectation, Conditional Expectation

$E[X;A]=E[X\bold1_A]$, $E[X|A] = \frac{E[X\bold1_A]}{P(A)}$

Partial expectation is the contribution of $A$ to $X$.

### Conditional Expectation

Given a random element $Y:\Omega\mapsto\mathcal Y$ and a random variable $X:\Omega\mapsto\R$

The $E^Y[X]=E[X|Y]$ is a random variable s.t.

1) $\exists g:Y\to\R$ s.t. $E[X|Y] = g(Y)$. Basically expectation of X|Y is a function of Y.
2) for all bounded $h:y\to\R$, $E[h(Y) | E[X|Y]]=E[h(Y)X]$

#### Thm

$E[X|Y]$ exists (in notes) and essentially unique.

If $g_1(y)$ and $g_2(y)$ satisfy 1, 2, then $P(g_1(y) !=g_2(y))=0$.

$$
h:y\to\{0,1\} \\
h(y) = \bold1_{g_1(Y)>g_2(Y)} \\
Eh(Y)g_1(Y) = Eh(Y)X=Eh(Y)g_2(Y) \\
E|g_1(Y) - g_2(Y)| = E[h(Y)(g_1(Y) - g_2(Y)) + (1 - h(Y))(g_2(Y) - g_1(Y))] \\
= 0 + E[g_2(Y) - g_1(Y)] - 0 \\
= 0
$$
$E[g_2(Y) - g_1(Y)]$ is 0 bc $\bold1:Y\to\{1\}$

Since $E|g_1(Y) - g_2(Y)|=0$, then it is unique.

$$
\{X_i\}_{i\sube\{1,2\}} \text{ be r.v.s have } E[X_i]<\infty \\
Y \text{ is y-valued r.e} \\
g:y\to\R,a\in\R,z:Y\to\Z
$$

1) linearity $E[X_1+aX_2 | Y]=E[X_1|Y] + aE[X_2|Y]$
2) monotone $X_1\ge X_2\implies E[X_1|Y]\geq E[X_2|Y]$
3) Tower Rule $E[E[X|Y]] = E[X]$
4) $E[g(Y)|Y] = g(Y)$, $E[E[X|Y]|Y] = E[X|Y]$
5) $E[E[X|Y]|Z(Y)]=E[X|Z(Y)]$
6) $E[E[X|Z(Y)]|Y] = E[X|Z(Y)]$
7) $X\perp Y, E[X|Y] = E[X]$
8) i) $x\in\R E[x|Y] = x$  
   ii) $y\in YE[X|y] = E[X]$
9) $W:\Omega\to\Omega, N(w)=w, E[X|W] = X$
10) $E[\empty(X) | Y]\ge E[X|Y]$
11) $(Y_n)_{n\in\N}\ge0, \nearrow X_\infty, E[Y_n|Y]\nearrow E[X_\infin|Y]$ monontone convergence
12) Fatou's Lemma is same thing but replace expectations with conditional expectations
13) Self Adjoint $E[(X_1)(E[X_2 | Y])] = E[(E[X_1|Y])(X_2)]$

Proof of linearity:

$g_1(Y)=E[X_1|Y], g_2(Y) = E[X_2|Y]$

claim $g_1 + ag_2=E(X_1+aX_2|Y)$

let $h:y\to\R$

$$
E[g_1+ag_2(Y)]h(Y) = Eg_1(Y)h(Y)+E[ag_2(Y)h(Y)]\\
= E[X_1]h(Y) + aE[X_2h(Y)] \\
= E[(X_1 + aX_2)h(Y)]
$$

## Markov Chains

The Markov property states the distribution of the future trajectory of the processes
given the past trajectory of the process only depends on the current state, and the
past does not matter.

### Probability Kernel

A probability kernel from $\mathcal X$ to $\mathcal Y$ is a function $k:\mathcal X\times\text{events}(\mathcal Y)\to[0,1]$ such that

1. For each $B\in\text{events}(\mathcal{Y})$ it holds that $k(\cdot,B)$ is an $\mathcal X$-
valued
2. For each $x\in\mathcal X$ it holds that $k(x,\cdot)$ is a probability measure on $\mathcal Y$.

We can interpret $k$ as a function that maps a fixed $x\in\mathcal X$ to a probability measure on $\mathcal Y$ and for a fixed $B\in\text{events}(\mathcal Y)$ as a random
variable on $\mathcal{X}$.

### Markov Transition Kernel

A markov transition kernel on $\mathcal X$ is a probability kernel from $\mathcal X$ to
$\mathcal X$. This may also be referred to a Markov kernel or a transition kernel.

### Stochastic Matrix

A stochastic matrix on $\mathcal{X}$ is an element, $p=(P(x,y))_{(x,y)\in\mathcal{X}^2}$ of
$\R^{\mathcal{X}^2}\simeq(\mathcal{X}\times\mathcal{X}\to\R)$ such that

1. $P(x,y)\ge 0$ for all $x,y\in\mathcal{X}$
2. \sum_{y\in\mathcal{X}}P(x,y)=1$ for all $x\in\mathcal{X}$

### Prop 7.1.5

If $\mathcal{X}$ is countable then there is a bijection between tranition kernels on
$\mathcal{X}$ and stochastic matrices on $\mathcal{X}$ given by

1. $k_P(x,A)=\delta_xP\textbf{1}_A=\sum_{y\in A}P(x,y)$
2. $P_k=(k(x,\{y\}))_{(x,y)\in\mathcal{X}^2}

