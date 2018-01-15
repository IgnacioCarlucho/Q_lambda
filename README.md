#Backwards view Q(λ)

The backwards view of TD(λ) implemented in Q-learning algorithm. 
Here I am using a custom gym environment, the non skid frozen lake, to be able to simplify the problem a bit. This is due to the fact that I am using tabular Q-learning, which struggles with high dimension spaces. 

## How to run: 

```
python Q_lambda.py 
```

This will give you an output of the type: 
```
('test', 1, 'lambda', 1.0, 'wins', 1416, 'losses', 584, 'efficiency', 242.47)
('test', 2, 'lambda', 0.9, 'wins', 1445, 'losses', 555, 'efficiency', 260.36)
('test', 3, 'lambda', 0.8, 'wins', 1417, 'losses', 583, 'efficiency', 243.05)
('test', 4, 'lambda', 0.7, 'wins', 1448, 'losses', 552, 'efficiency', 262.32)
('test', 5, 'lambda', 0.6, 'wins', 1437, 'losses', 563, 'efficiency', 255.24)
('test', 6, 'lambda', 0.5, 'wins', 1418, 'losses', 582, 'efficiency', 243.64)
('test', 7, 'lambda', 0.4, 'wins', 1464, 'losses', 536, 'efficiency', 273.13)
('test', 8, 'lambda', 0.3, 'wins', 1449, 'losses', 551, 'efficiency', 262.98)
('test', 9, 'lambda', 0.2, 'wins', 1447, 'losses', 553, 'efficiency', 261.66)
('test', 10, 'lambda', 0.1, 'wins', 1430, 'losses', 570, 'efficiency', 250.88)
('test', 11, 'lambda', 0.0, 'wins', 1436, 'losses', 564, 'efficiency', 254.61)
```
Here in each test I do 2000 runs of the episode. Each test uses a different value of λ (in the range 1 to 0). This evaluation is done to obtain the optimal lambda value. Of course sucesive runs and a statistical analysis should be performed to obtain this ( similar to what barto does on his book). I can say that after a few runs λ = 0.4 shows as an optimal 