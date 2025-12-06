import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JUSTICES_CSV = os.path.join(SCRIPT_DIR, 'justices.csv')

_ideology_cache = None

def load_ideology_data():
    global _ideology_cache
    if _ideology_cache is None:
        _ideology_cache = pd.read_csv(JUSTICES_CSV)
        _ideology_cache = _ideology_cache[['term', 'justiceName', 'post_med']]
    return _ideology_cache


def get_justice_ideology(justice_name, term):
    df = load_ideology_data()
    match = df[(df['justiceName'] == justice_name) & (df['term'] == term)]

    if len(match) == 0:
        justice_scores = df[df['justiceName'] == justice_name]
        if len(justice_scores) > 0:
            closest_term = justice_scores.iloc[(justice_scores['term'] - term).abs().argsort()[:1]]
            return closest_term['post_med'].values[0]
        return 0.0

    return match['post_med'].values[0]


def compute_median_court_ideology(term):
    df = load_ideology_data()
    term_justices = df[df['term'] == term]

    if len(term_justices) == 0:
        all_terms = df['term'].unique()
        nearest_term = all_terms[np.argmin(np.abs(all_terms - term))]
        term_justices = df[df['term'] == nearest_term]

    if len(term_justices) == 0:
        return 0.0

    return term_justices['post_med'].median()


def add_ideology_features(df):
    df = df.copy()

    print("Computing median court ideology for each term using real MQ scores...")
    df['median_ideology'] = df['term'].apply(compute_median_court_ideology)

    df['conservative_era'] = (df['term'] >= 2017).astype(int)
    df['liberal_era'] = (df['term'] < 1986).astype(int)

    print(f"\nIdeology feature statistics:")
    print(f"  median_ideology range: [{df['median_ideology'].min():.3f}, {df['median_ideology'].max():.3f}]")
    print(f"  median_ideology mean: {df['median_ideology'].mean():.3f}")
    print(f"  conservative_era cases: {df['conservative_era'].sum()} ({100*df['conservative_era'].mean():.1f}%)")
    print(f"  liberal_era cases: {df['liberal_era'].sum()} ({100*df['liberal_era'].mean():.1f}%)")

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Real Martin-Quinn Ideology Features")
    print("=" * 60)

    ideology_df = load_ideology_data()
    print(f"\nLoaded {len(ideology_df)} justice-term records")
    print(f"Terms covered: {ideology_df['term'].min()} - {ideology_df['term'].max()}")
    print(f"Unique justices: {ideology_df['justiceName'].nunique()}")

    print("\n" + "=" * 60)
    print("Median Court Ideology by Era (Real MQ Scores)")
    print("=" * 60)

    test_terms = [
        (1960, "Warren Court (liberal)"),
        (1980, "Burger Court (moderate)"),
        (1995, "Rehnquist Court (conservative)"),
        (2010, "Roberts + Kennedy (moderate-conservative)"),
        (2017, "Roberts + Gorsuch (conservative)"),
        (2019, "Roberts + Kavanaugh (conservative)"),
        (2021, "Roberts + Barrett (very conservative)"),
        (2023, "Current Court"),
    ]

    for term, description in test_terms:
        median_ideology = compute_median_court_ideology(term)
        label = "CONSERVATIVE" if median_ideology > 0 else "LIBERAL"
        print(f"{term} ({description:35s}): {median_ideology:6.3f} ({label})")

    print("\n" + "=" * 60)
    print("Hardcoded vs Real Scores Comparison")
    print("=" * 60)

    hardcoded = {
        2023: 0.50,
        2020: 0.50,
        2019: 0.40,
        2018: 0.40,
        2017: 0.35,
        2010: 0.20,
        2000: 0.60,
        1990: 0.60,
    }

    print(f"{'Term':<6} {'Hardcoded':<12} {'Real MQ':<12} {'Difference':<12}")
    print("-" * 50)
    for term in sorted(hardcoded.keys(), reverse=True):
        real = compute_median_court_ideology(term)
        diff = real - hardcoded[term]
        print(f"{term:<6} {hardcoded[term]:<12.3f} {real:<12.3f} {diff:+.3f}")

    print("\n✓ Real Martin-Quinn scores loaded successfully!")
