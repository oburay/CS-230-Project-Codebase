"""
Oyez Transcript Scraper for SCOTUS Cases
Fetches oral argument transcripts from Oyez API for cases in SCDB dataset
"""

import pandas as pd
import requests
import time
import json
from pathlib import Path
from typing import Optional, Dict, List
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_oral_argument(term: int, docket: str) -> Optional[Dict]:
    """
    Fetch oral argument transcript from Oyez API

    Args:
        term: Supreme Court term year (e.g., 2023)
        docket: Docket number (e.g., "22-800")

    Returns:
        Dictionary with transcript text and metadata, or None if unavailable
    """
    try:
        # Get case information
        case_url = f"https://api.oyez.org/cases/{term}/{docket}"
        response = requests.get(case_url, timeout=30)

        if response.status_code != 200:
            logger.debug(f"Case not found: {term}/{docket}")
            return None

        data = response.json()

        # Check if oral argument exists
        if 'oral_argument_audio' not in data or not data['oral_argument_audio']:
            logger.debug(f"No oral argument for: {term}/{docket}")
            return None

        # Get oral argument details
        arg_href = data['oral_argument_audio'][0]['href']
        arg_response = requests.get(arg_href, timeout=30)

        if arg_response.status_code != 200:
            logger.debug(f"Could not fetch oral argument: {term}/{docket}")
            return None

        arg_data = arg_response.json()

        # Check if transcript exists
        if 'transcript' not in arg_data or not arg_data['transcript']:
            logger.debug(f"No transcript for: {term}/{docket}")
            return None

        transcript = arg_data['transcript']
        sections = transcript.get('sections', [])

        if not sections:
            logger.debug(f"Empty transcript for: {term}/{docket}")
            return None

        # Extract full transcript text
        full_text = []
        turn_count = 0

        for section in sections:
            for turn in section.get('turns', []):
                speaker_name = turn.get('speaker', {}).get('name', 'Unknown')
                text_blocks = turn.get('text_blocks', [])

                for block in text_blocks:
                    text = block.get('text', '').strip()
                    if text:
                        full_text.append(f"{speaker_name}: {text}")
                        turn_count += 1

        if not full_text:
            logger.debug(f"No text content in transcript: {term}/{docket}")
            return None

        transcript_text = "\n".join(full_text)

        return {
            'term': term,
            'docket': docket,
            'transcript': transcript_text,
            'turn_count': turn_count,
            'char_count': len(transcript_text),
            'case_name': data.get('name', 'Unknown')
        }

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout for {term}/{docket}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Request error for {term}/{docket}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error for {term}/{docket}: {str(e)}")
        return None


def scrape_oyez_transcripts(
    scdb_path: str,
    output_file: str = 'oyez_transcripts.json',
    start_year: int = 2000,
    end_year: Optional[int] = None,
    rate_limit: float = 1.0
) -> pd.DataFrame:
    """
    Scrape Oyez transcripts for cases in SCDB dataset

    Args:
        scdb_path: Path to SCDB CSV file
        output_file: Output JSON file path
        start_year: Start year for scraping (default: 2000)
        end_year: End year for scraping (default: None = all years)
        rate_limit: Seconds to wait between requests (default: 1.0)

    Returns:
        DataFrame with transcripts and metadata
    """
    logger.info(f"Loading SCDB data from {scdb_path}")
    scdb = pd.read_csv(scdb_path, encoding='latin1')

    # Filter by year range
    mask = scdb['term'] >= start_year
    if end_year:
        mask &= scdb['term'] <= end_year

    filtered_scdb = scdb[mask].copy()
    logger.info(f"Found {len(filtered_scdb)} cases between {start_year} and {end_year or 'present'}")

    transcripts = []
    success_count = 0
    fail_count = 0

    for idx, row in filtered_scdb.iterrows():
        term = row['term']
        docket = str(row['docket'])
        case_name = row['caseName']

        logger.info(f"[{idx+1}/{len(filtered_scdb)}] Fetching: {case_name} ({term}/{docket})")

        # Fetch transcript
        result = get_oral_argument(term, docket)

        if result:
            # Add SCDB metadata
            result['caseId'] = row['caseId']
            result['caseName'] = case_name
            result['decisionDirection'] = row['decisionDirection']

            # Add other relevant SCDB features
            for col in ['chief', 'issue', 'issueArea', 'petitioner', 'respondent',
                       'jurisdiction', 'caseOrigin', 'caseSource', 'lcDisposition',
                       'decisionType', 'majVotes', 'minVotes']:
                if col in row.index:
                    result[col] = row[col]

            transcripts.append(result)
            success_count += 1
            logger.info(f"✅ Success ({result['turn_count']} turns, {result['char_count']:,} chars)")
        else:
            fail_count += 1
            logger.info(f"⚠️  No transcript available")

        # Save progress after each successful transcript (to avoid data loss)
        if transcripts:
            temp_df = pd.DataFrame(transcripts)
            temp_path = f"{output_file}.temp"
            temp_df.to_json(temp_path, orient='records', indent=2)

            # Log every 10 transcripts
            if len(transcripts) % 10 == 0:
                logger.info(f"📊 Progress saved: {len(transcripts)} transcripts")

        # Rate limiting
        time.sleep(rate_limit)

    # Create final DataFrame
    if transcripts:
        df = pd.DataFrame(transcripts)
        df.to_json(output_file, orient='records', indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Scraping complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Total cases processed: {len(filtered_scdb)}")
        logger.info(f"Successful scrapes: {success_count} ({success_count/len(filtered_scdb)*100:.1f}%)")
        logger.info(f"Failed scrapes: {fail_count}")
        logger.info(f"Average transcript length: {df['char_count'].mean():,.0f} characters")
        logger.info(f"Average turns per case: {df['turn_count'].mean():.0f}")
        logger.info(f"Output saved to: {output_file}")
        logger.info(f"{'='*60}\n")

        return df
    else:
        logger.warning("No transcripts found!")
        return pd.DataFrame()


if __name__ == "__main__":
    # Configuration
    SCDB_PATH = "SCDB_2025_01_caseCentered_Citation.csv"
    OUTPUT_PATH = "oyez_transcripts.json"
    START_YEAR = 2000
    RATE_LIMIT = 1.5  # Be nice to Oyez API

    # Run scraper
    logger.info("Starting Oyez transcript scraper...")
    logger.info(f"Target: Cases from {START_YEAR} onwards")
    logger.info(f"Rate limit: {RATE_LIMIT}s per request\n")

    df = scrape_oyez_transcripts(
        scdb_path=SCDB_PATH,
        output_file=OUTPUT_PATH,
        start_year=START_YEAR,
        rate_limit=RATE_LIMIT
    )

    if not df.empty:
        # Show sample statistics
        logger.info("Sample transcript statistics:")
        logger.info(f"Shortest: {df['char_count'].min():,} chars")
        logger.info(f"Longest: {df['char_count'].max():,} chars")
        logger.info(f"Median: {df['char_count'].median():,.0f} chars")

        # Show year distribution
        logger.info("\nTranscripts by year:")
        year_counts = df['term'].value_counts().sort_index()
        for year, count in year_counts.items():
            logger.info(f"  {year}: {count} cases")
