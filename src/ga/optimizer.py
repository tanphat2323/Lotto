import numpy as np
import random
from copy import deepcopy

class WheelOptimizer:
    def __init__(self,
                 model_probs_main,
                 model_probs_special,
                 pop_size=50,
                 generations=20,
                 ticket_budget=10,
                 main_nums=35,
                 pick_size=5):
        """
        GA Optimizer for Lotto Wheels.
        model_probs_main: array of shape (35,) - prob of each number
        """
        self.probs_main = model_probs_main
        self.probs_special = model_probs_special
        self.pop_size = pop_size
        self.generations = generations
        self.ticket_budget = ticket_budget
        self.main_nums = main_nums
        self.pick_size = pick_size

    def _create_individual(self):
        """
        An individual is a set of `ticket_budget` tickets.
        Each ticket is a set of 5 numbers.
        """
        tickets = []
        for _ in range(self.ticket_budget):
            # Weighted random sample based on LSTM probs?
            # Or just random? Better to bias initialization with high probs.

            # Normalize probs to sum to 1 for sampling
            p = self.probs_main / self.probs_main.sum()

            # Sample without replacement
            ticket = np.random.choice(range(1, self.main_nums+1), size=self.pick_size, replace=False, p=p)
            tickets.append(sorted(list(ticket)))
        return tickets

    def _fitness(self, individual):
        """
        Fitness = Sum of log probs of all numbers in all tickets (Coverage)
                 + Diversity bonus (optional)
                 - Cost (fixed here by budget)

        Simple version: Sum of probabilities of numbers in the set (Union).
        If we buy a set of tickets, we want to maximize the probability that *at least one* wins?
        Or expected return?

        Let's maximize Expected Return approx = Sum of probabilities of picked numbers.
        But simply summing probs favors picking the same high-prob numbers repeatedly.
        We want coverage.

        Score = Sum(Prob(n) for n in Union(tickets))
        This encourages picking *different* high prob numbers.
        """
        # Flatten unique numbers covered by this wheel
        unique_numbers = set()
        for ticket in individual:
            unique_numbers.update(ticket)

        score = 0
        for n in unique_numbers:
            score += self.probs_main[n-1] # 0-indexed prob

        return score

    def _crossover(self, parent1, parent2):
        # Swap tickets between wheels
        cut = random.randint(1, self.ticket_budget - 1)
        child1 = parent1[:cut] + parent2[cut:]
        child2 = parent2[:cut] + parent1[cut:]
        return child1, child2

    def _mutate(self, individual):
        # Mutate a ticket in the wheel
        if random.random() < 0.2:
            idx = random.randint(0, self.ticket_budget - 1)
            # Re-sample a ticket
            p = self.probs_main / self.probs_main.sum()
            new_ticket = np.random.choice(range(1, self.main_nums+1), size=self.pick_size, replace=False, p=p)
            individual[idx] = sorted(list(new_ticket))
        return individual

    def optimize(self):
        population = [self._create_individual() for _ in range(self.pop_size)]

        best_fitness = -1
        best_ind = None

        for g in range(self.generations):
            # Evaluate
            scores = [(ind, self._fitness(ind)) for ind in population]
            scores.sort(key=lambda x: x[1], reverse=True)

            if scores[0][1] > best_fitness:
                best_fitness = scores[0][1]
                best_ind = deepcopy(scores[0][0])

            # Selection (Top 50%)
            survivors = [s[0] for s in scores[:self.pop_size//2]]

            # Repopulate
            new_pop = survivors[:]
            while len(new_pop) < self.pop_size:
                p1 = random.choice(survivors)
                p2 = random.choice(survivors)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))

            population = new_pop

        return best_ind, best_fitness

if __name__ == "__main__":
    # Test GA
    probs = np.random.rand(35)
    probs = probs / probs.sum()

    ga = WheelOptimizer(probs, None, ticket_budget=5)
    best_wheel, score = ga.optimize()
    print(f"Best Score: {score}")
    print("Best Wheel:")
    for t in best_wheel:
        print(t)
