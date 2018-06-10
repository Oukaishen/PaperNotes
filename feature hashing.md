## feature hashing 

created by kaishen, 10 Jun

<br>

Also known as "hash trick". The idea behind this name simple: **convert the data into a vector of features. **

Similar methods are like "one-hot encode", "embedding".  

<br>

Taking an example like this: 

If there is a text "Hi, kaishen is using hashing trick" and we want to convert it to a vector. 

**First** thing is to determine how large the dimension you want in the output vector. 

In general, this number can be VERY large, up to 2**25. Let say we use 5 in this example.

<br>

**Second** we can use any good hash function that accept a word and output a number in [0,4].

Let say:

hash(Hi) = 0, hash(kaishen) = 1, hash(is) = 1, hash(using) = 2, hash(hashing) = 3, hash(trick) = 4,

Then the final vector is [1,2,1,1,1]

Notice that we just add 1 to the i-th dimension of ther vector each time our hash function returns that dimension. Sometimes you can use a 2nd hash function that return {-1, +1} to determine whether it is add or substract.

<br>

**[Pro]** There is one advantage for hash trick compared to the "one-hot" method. It is Friendly to online learning method where you can train on a dataset that doesn't fit in memory because you need to see each example only once. One-hot encoding will not work well with online learning beacuse to prepare dictionaries you need to see whole dataset first.

**[Con]** feature hashing does not have good interpretation.

