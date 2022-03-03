# Find-S Algorithm implemented in python

Referring to enjoySport problem stated in Tom M. Mitchell's Machine learning book.
To implement the find-s algorithm we need to have a generalized hypothesis using the find-s 
algorithm.

the algorithm is stated as:
```
1. Initialize h to the most specific hypothesis in H
2. For each positive training instance x
    . For each attribute constraint ai in h
        If the constraint ai is satistfied by x
            Do nothing
        else
            Replace ai in h by the next more general constraint that is satisfied by x
            
3. Output hypothesis h
```


The data used is csv file shown bellow:

|Sky|AirTemp|Humidity|Wind|Water|Forecast|EnjoySport|
|---|-------|--------|----|-----|---------|---------|
|Sunny|Warm|Normal|Strong|Warm|Same|Yes|
|Sunny|Warm|High|Strong|Warm|Same|Yes|
|Rainy|Cold|High|Strong|Warm|Change|No|
|Sunny|Warm|High|Strong|Cold|Change|Yes|

### The output of hypothesis

* h0 = <Φ, Φ, Φ, Φ, Φ, Φ>
* h1 = <Sunny, Warm, Normal, Strong, Warm, Same>
* h2 = <Sunny, Warm, ?, Strong, Warm, Same>
* h3 = <Sunny, Warm, ?, Strong, Warm, Same>
* h4 = <Sunny, Warm, ?, Strong, ?,?>


We can see the last hypothesis as the more generalized one which is to be returned after the training process.