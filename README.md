# Backwards view Q(λ)

The backwards view of TD(λ) implemented in Q-learning algorithm. 
Here I am using a custom gym environment, the non skid frozen lake, to be able to simplify the problem a bit. This is due to the fact that I am using tabular Q-learning, which struggles with high dimension spaces. If the variable vanilla_Q is set to false, the watkins Q lambda algorithm is implemented. Otherwise the vanilla Q lambda is implemented (called naive in suttons book). In this version the traces are not set to zero every time a random action is taken, but on the contrary, decreased (it showed worst performance than that of watkins).

## How to run: 

```
python Q_lambda.py 
```

This will give you an output of the type: 
```
('test', 1, 'lambda', 1.0, 'wins', 2347, 'losses', 653, 'efficiency', 78.23)
('test', 2, 'lambda', 0.9, 'wins', 2299, 'losses', 701, 'efficiency', 76.63)
('test', 3, 'lambda', 0.8, 'wins', 2308, 'losses', 692, 'efficiency', 76.93)
('test', 4, 'lambda', 0.7, 'wins', 0, 'losses', 3000, 'efficiency', 0.0)
('test', 5, 'lambda', 0.6, 'wins', 2317, 'losses', 683, 'efficiency', 77.23)
('test', 6, 'lambda', 0.5, 'wins', 2249, 'losses', 751, 'efficiency', 74.97)
('test', 7, 'lambda', 0.4, 'wins', 2351, 'losses', 649, 'efficiency', 78.37)
('test', 8, 'lambda', 0.3, 'wins', 2182, 'losses', 818, 'efficiency', 72.73)
('test', 9, 'lambda', 0.2, 'wins', 2325, 'losses', 675, 'efficiency', 77.5)
('test', 10, 'lambda', 0.1, 'wins', 2281, 'losses', 719, 'efficiency', 76.03)
('test', 11, 'lambda', 0.0, 'wins', 2336, 'losses', 664, 'efficiency', 77.87)

```
Here in each test I do 2000 runs of the episode. Each test uses a different value of λ (in the range 1 to 0). This evaluation is done to obtain the optimal lambda value. Of course sucesive runs and a statistical analysis should be performed to obtain this ( similar to what barto does on his book). I can say that after a few runs λ = 0.4 shows as an optimal 