"""
Justice ideology scores for SCOTUS prediction.

Martin-Quinn scores measure justice ideology on liberal-conservative spectrum.
Negative = liberal, Positive = conservative

Source: Martin-Quinn Database (http://mqscores.lsa.umich.edu/)
Using real term-specific scores from justices.csv
"""

import pandas as pd
import numpy as np
import os

# Load real Martin-Quinn scores
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JUSTICES_CSV = os.path.join(SCRIPT_DIR, 'justices.csv')

# Load and cache the data
_ideology_cache = None

def load_ideology_data():
    """Load Martin-Quinn ideology scores from CSV."""
    global _ideology_cache
    if _ideology_cache is None:
        _ideology_cache = pd.read_csv(JUSTICES_CSV)
        # Use post_med (posterior median) as the ideology score
        _ideology_cache = _ideology_cache[['term', 'justiceName', 'post_med']]
    return _ideology_cache


def get_justice_ideology(justice_name, term):
    """
    Get ideology score for a specific justice in a specific term.

    Args:
        justice_name: Justice name (e.g., 'JGRoberts')
        term: Court term year

    Returns:
        float: Ideology score (negative=liberal, positive=conservative)
    """
    df = load_ideology_data()
    match = df[(df['justiceName'] == justice_name) & (df['term'] == term)]

    if len(match) == 0:
        # If no exact match, try to find closest term for that justice
        justice_scores = df[df['justiceName'] == justice_name]
        if len(justice_scores) > 0:
            # Use closest available term
            closest_term = justice_scores.iloc[(justice_scores['term'] - term).abs().argsort()[:1]]
            return closest_term['post_med'].values[0]
        return 0.0  # Default to moderate if justice not found

    return match['post_med'].values[0]


def compute_median_court_ideology(term):
    """
    Compute median ideology of the Supreme Court for a given term.

    This represents the ideological center of the court that year.
    Uses all 9 justices serving in that term.

    Args:
        term: Court term year

    Returns:
        float: Median ideology score for that term's court
    """
    df = load_ideology_data()
    term_justices = df[df['term'] == term]

    if len(term_justices) == 0:
        # If no data for this term, use nearest available term
        all_terms = df['term'].unique()
        nearest_term = all_terms[np.argmin(np.abs(all_terms - term))]
        term_justices = df[df['term'] == nearest_term]

    if len(term_justices) == 0:
        return 0.0  # Fallback to moderate

    # Return median ideology score
    return term_justices['post_med'].median()


def add_ideology_features(df):
    """
    Add justice ideology features to dataframe.

    Features added:
    - median_ideology: Median ideology of court in that term (real MQ scores)
    - conservative_era: Binary flag for post-2017 (conservative majority solidified)
    - liberal_era: Binary flag for pre-1986 (Warren/Burger liberal courts)

    Args:
        df: DataFrame with 'term' column

    Returns:
        DataFrame with added ideology features
    """
    df = df.copy()

    # Add median court ideology by term using REAL Martin-Quinn scores
    print("Computing median court ideology for each term using real MQ scores...")
    df['median_ideology'] = df['term'].apply(compute_median_court_ideology)

    # Add court composition indicators (same as before)
    df['conservative_era'] = (df['term'] >= 2017).astype(int)  # Post-Gorsuch
    df['liberal_era'] = (df['term'] < 1986).astype(int)  # Pre-Rehnquist

    # Print summary statistics
    print(f"\nIdeology feature statistics:")
    print(f"  median_ideology range: [{df['median_ideology'].min():.3f}, {df['median_ideology'].max():.3f}]")
    print(f"  median_ideology mean: {df['median_ideology'].mean():.3f}")
    print(f"  conservative_era cases: {df['conservative_era'].sum()} ({100*df['conservative_era'].mean():.1f}%)")
    print(f"  liberal_era cases: {df['liberal_era'].sum()} ({100*df['liberal_era'].mean():.1f}%)")

    return df


if __name__ == "__main__":
    # Test the ideology computation
    print("=" * 60)
    print("Testing Real Martin-Quinn Ideology Features")
    print("=" * 60)

    # Load data
    ideology_df = load_ideology_data()
    print(f"\nLoaded {len(ideology_df)} justice-term records")
    print(f"Terms covered: {ideology_df['term'].min()} - {ideology_df['term'].max()}")
    print(f"Unique justices: {ideology_df['justiceName'].nunique()}")

    # Show median ideology over time for key eras
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

    # Compare old hardcoded vs new real scores
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
