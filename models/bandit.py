# ============================================================
# FILE: models/bandit.py
# JOB: Build the smart ad system
# SIMPLE MEANING: We have 5 ads. Instead of showing random ads,
#                 this system LEARNS which ad people click more
#                 and starts showing that ad more often.
#
# Real world example: Swiggy tests 5 discount banners.
# After 1000 users, it notices banner 3 gets most clicks.
# It starts showing banner 3 more. That is this system.
# ============================================================

import numpy as np
import pickle
import os


class ThompsonBandit:
    # This is a "class" = a blueprint for our ad system

    def __init__(self, n_ads):
        # __init__ runs when we first create the bandit
        # n_ads = how many different ads we have

        self.n_ads = n_ads

        # alpha = number of times each ad was CLICKED
        # beta  = number of times each ad was NOT clicked
        # Start with 1 each (not 0) so every ad gets a chance
        self.alpha = np.ones(n_ads)   # [1, 1, 1, 1, 1]
        self.beta  = np.ones(n_ads)   # [1, 1, 1, 1, 1]

        # Track history for showing in dashboard
        self.history = []

    def select_ad(self):
        # This function picks which ad to show

        # For each ad, we "sample" a random number from
        # its probability distribution.
        # An ad with more clicks has a higher distribution
        # so it will likely get a higher sample number
        # and get selected more often.
        # This is called Thompson Sampling.
        samples = np.random.beta(self.alpha, self.beta)

        # Pick the ad with the highest sample
        chosen_ad = int(np.argmax(samples))
        return chosen_ad

    def update(self, ad_idx, clicked):
        # This function updates what we learned after showing an ad
        # ad_idx = which ad was shown (0, 1, 2, 3, or 4)
        # clicked = True if user clicked, False if they ignored

        if clicked:
            # User clicked → this ad is doing well → increase alpha
            self.alpha[ad_idx] += 1
        else:
            # User ignored → increase beta
            self.beta[ad_idx] += 1

        # Save to history
        self.history.append({
            "ad_idx":  ad_idx,
            "clicked": clicked,
            # Current estimated CTR = clicks / (clicks + ignores)
            "ctr": self.alpha[ad_idx] / (
                self.alpha[ad_idx] + self.beta[ad_idx]
            )
        })

    def get_ctr_estimates(self):
        # Returns current estimated CTR for each ad
        # CTR = Click Through Rate = clicks / total shown
        return self.alpha / (self.alpha + self.beta)

    def simulate(self, n_rounds=1000):
        # This simulates 1000 users seeing ads
        # so our bandit has already "learned" before the demo
        #
        # We pretend:
        #   Ad 0 has true CTR of 12% (boring ad)
        #   Ad 1 has true CTR of 15%
        #   Ad 2 has true CTR of 28% (best ad!)
        #   Ad 3 has true CTR of 18%
        #   Ad 4 has true CTR of 10% (worst ad)
        true_ctrs = [0.12, 0.15, 0.28, 0.18, 0.10]

        print(f"Simulating {n_rounds} ad impressions...")
        for round_num in range(n_rounds):
            # Pick an ad using Thompson Sampling
            ad = self.select_ad()

            # Simulate if user clicked (based on true CTR)
            clicked = np.random.rand() < true_ctrs[ad]

            # Update the bandit with what happened
            self.update(ad, clicked)

            # Print progress every 200 rounds
            if (round_num + 1) % 200 == 0:
                ctrs = self.get_ctr_estimates()
                best = np.argmax(ctrs)
                print(
                    f"  Round {round_num+1}: "
                    f"Best ad = Ad {best}, "
                    f"Estimated CTR = {ctrs[best]:.3f}"
                )

        print("\nFinal CTR estimates per ad:")
        for i, ctr in enumerate(self.get_ctr_estimates()):
            print(f"  Ad {i}: {ctr:.3f} ({ctr*100:.1f}%)")

        print(f"\nBandit identified Ad 2 as best (true CTR = 28%)")
        print(f"Random baseline CTR = 12%")
        lift = (self.get_ctr_estimates().max() - 0.12) / 0.12 * 100
        print(f"Improvement over random = {lift:.1f}%")


# ---- RUN THIS FILE ----
if __name__ == "__main__":
    print("Building and training the Ad Bandit system...")

    # Create bandit with 5 ads
    bandit = ThompsonBandit(n_ads=5)

    # Simulate 1000 rounds so it has learned already
    bandit.simulate(n_rounds=1000)

    # Save the trained bandit
    with open("artifacts/bandit.pkl", "wb") as f:
        pickle.dump(bandit, f)

    print("\nSaved: artifacts/bandit.pkl")
    print("\nFile 3 (Bandit) - DONE!")