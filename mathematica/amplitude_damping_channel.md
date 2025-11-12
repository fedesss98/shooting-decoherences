# Amplitude Damping Channel

### Prove that it forms a Semigroup

A map (channel) is a Semigroup if $![0ln45ij2k2xo9](img/0ln45ij2k2xo9.png)$.
So, if $![0akkba2lt8ddu](img/0akkba2lt8ddu.png)$, we have:
    $![1g6lhm95bdazv](img/1g6lhm95bdazv.png)$, and that should be equal to $![14jtysailp4a6](img/14jtysailp4a6.png)$.
In this case the map is given in terms of its Kraus Operators $![0gvyfo7rty6ug](img/0gvyfo7rty6ug.png)$.

![1porv23c8cn91](img/1porv23c8cn91.png)

|  |  |
| - | - |
| 1 | 0 |
| 0 | -Power- |

|  |  |
| - | - |
| 0 | -Power- |
| 0 | 0 |

```wl
In[]:= \[CapitalLambda][x_, t_] := Module[{k1t, k2t, \[Rho]t}, 
    k1t = k1 /. \[Tau] -> t; 
    k2t = k2 /. \[Tau] -> t; 
    \[Rho]t = k1t . x . ConjugateTranspose[k1t] + k2t . x . ConjugateTranspose[k2t]; 
    Simplify[\[Rho]t, Assumptions -> {\[CapitalGamma] > 0, t > 0 }] 
   ]
```

![0hfw810pta00o](img/0hfw810pta00o.png)

```wl
In[]:= \[Rho]\[Tau] = \[CapitalLambda][\[Rho], \[Tau]];
 \[Rho]\[Tau] // MatrixForm
```

|  |  |
| - | - |
| -Plus- | -Times- |
| -Times- | -Times- |

First we compute $![0b69w05lrme9p](img/0b69w05lrme9p.png)$,

```wl
In[]:= \[Rho]s = \[CapitalLambda][\[Rho], s];
 \[Rho]s // MatrixForm
```

|  |  |
| - | - |
| -Plus- | -Times- |
| -Times- | -Times- |

Then $![1emul94egtzfi](img/1emul94egtzfi.png)$,

```wl
In[]:= \[Rho]ts = \[CapitalLambda][\[Rho]s, t];
 \[Rho]ts // MatrixForm
```

|  |  |
| - | - |
| -Plus- | -Times- |
| -Times- | -Times- |

Now we check that this is equal to $![1r4o09bb3rsid](img/1r4o09bb3rsid.png)$,

```wl
In[]:= \[Rho]ts == \[CapitalLambda][\[Rho], t + s]
```

```wl
Out[]= True
```

### Petz Recovery Map

The recovery map is defined as: $![07s8y4jtvh77v](img/07s8y4jtvh77v.png)$, where $![0rdfi6ackpigq](img/0rdfi6ackpigq.png)$, such that:
    $![1kydrr87iyxs5](img/1kydrr87iyxs5.png)$
(we reversed the time evolution from $![1wrqr2ailbtf1](img/1wrqr2ailbtf1.png)$ to $![0twal053oyrhw](img/0twal053oyrhw.png)$.

```wl
In[]:= Petz[Ks_, t_, \[Sigma]_, \[Rho]_] := Module[{innerMatrix, Phi, PhiAdj, recovered}, 
    Phi[rho_] := Total[(# . rho . ConjugateTranspose[#]) & /@ Ks /. \[Tau] -> t]; 
    PhiAdj[X_] := Total[(ConjugateTranspose[#] . X . #) & /@ Ks /. \[Tau] -> t]; 
    innerMatrix = (Phi[\[Sigma]])^(-1/2) . \[Rho] . (Phi[\[Sigma]])^(-1/2); (*TODO: Use the correct Power Operator for matrices (this takes the sqrt element by element)*)
    recovered = \[Sigma]^(1/2) . PhiAdj[innerMatrix] . \[Sigma]^(1/2); 
    FullSimplify[recovered, Assumptions -> {\[CapitalGamma] > 0, t > 0 }] 
   ]
```