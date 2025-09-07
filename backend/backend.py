"""
Legal Document Analysis Backend
Designed specifically for Google Gemini API integration
"""

import os
import json
import re
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration constants
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 2000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-pro-latest')

# Initialize Gemini
try:
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    logger.info("Google Gemini API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")


class DocumentAnalyzer:
    """
    Main class for analyzing legal documents using Google Gemini
    """

    def __init__(self):
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.max_retries = 3

    def create_chunks(self, full_text: str):
        """
        Split text into overlapping chunks, respecting sentence boundaries
        """
        # Split by sentence-ending punctuation while keeping delimiters
        sentences = re.split(r'(?<=[.!?])\s+', full_text)

        chunks = []
        current_chunk_words = []

        for sentence in sentences:
            words = sentence.split()
            if len(current_chunk_words) + len(words) > CHUNK_SIZE:
                # Finalize current chunk
                if current_chunk_words:
                    chunks.append(" ".join(current_chunk_words))

                # Start new chunk with overlap
                overlap_words = current_chunk_words[-CHUNK_OVERLAP:] if len(
                    current_chunk_words) >= CHUNK_OVERLAP else current_chunk_words
                current_chunk_words = overlap_words + words
            else:
                current_chunk_words.extend(words)

        # Add the last chunk
        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))

        logger.info(f"Document split into {len(chunks)} chunks")
        return chunks

    def get_analysis_prompt(self, document_text: str) -> str:
        """
        Create the detailed prompt for Gemini analysis
        """
        return f"""You are a methodical, high-precision data extraction engine that follows a strict two-step verification process before producing an answer. Your goal is to convert the provided legal document into a structured JSON object.

### YOUR INTERNAL PROCESS ###
1. **Draft Extraction**: First, mentally perform an initial extraction of the data based on the rules and schema. Do not output this draft.
2. **Self-Verification**: Next, critically review your draft against the checklist below. If you find any errors, correct them.
   * **JSON Validity Check**: Is the draft a single, raw, minified JSON object with no extra text or markdown?
   * **Schema Compliance Check**: Does the draft have exactly the four required keys: "key_parties", "critical_dates", "party_obligations", "red_flags"?
   * **Grounding Check**: Is every single value in the draft directly supported by the source text, with no invented information?
   * **Completeness Check**: Are categories with no information correctly represented as an empty array `[]`?
3. **Final Output**: After your verification is complete, provide the final, corrected JSON object as your response.

Your entire response MUST consist of ONLY the final, verified JSON object.

### RULES & SCHEMA ###
* **key_parties**: Array of objects, each with "name" and "role".
* **critical_dates**: Array of objects, each with "event" and "date".
* **party_obligations**: An object where each key is a party's name, and the value is an array of their obligation strings.
* **red_flags**: Array of strings describing risky or non-standard clauses.

### EXAMPLE ###
**Input Text:** "This Freelance Agreement ('Agreement'), effective Oct 1, 2025, is between Innovate Inc. ('Client') and Jane Doe ('Freelancer'). Freelancer agrees to deliver the project by Nov 15, 2025. This contract is governed by the laws of California."
**Expected JSON Output:**
{{"key_parties":[{{"name":"Innovate Inc.","role":"Client"}},{{"name":"Jane Doe","role":"Freelancer"}}],"critical_dates":[{{"event":"Effective Date","date":"Oct 1, 2025"}},{{"event":"Project Delivery Deadline","date":"Nov 15, 2025"}}],"party_obligations":{{"Jane Doe":["Deliver the project by Nov 15, 2025"]}},"red_flags":["The governing law is specified as California, which may have specific legal implications."]}}

### DOCUMENT FOR ANALYSIS ###
---
{document_text}
---"""

    def clean_gemini_response(self, raw_text: str) -> str:
        """
        Clean Gemini's response to extract pure JSON
        """
        # Remove markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()

        # Remove any leading/trailing text
        lines = raw_text.strip().split('\n')
        json_start = -1
        json_end = -1

        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                json_start = i
                break

        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().endswith('}'):
                json_end = i
                break

        if json_start != -1 and json_end != -1:
            return '\n'.join(lines[json_start:json_end + 1])

        return raw_text.strip()

    def analyze_chunk(self, chunk_text: str):
        """
        Analyze a single chunk using Gemini
        """
        prompt = self.get_analysis_prompt(chunk_text)

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Analyzing chunk (attempt {attempt + 1})")
                response = self.model.generate_content(prompt)

                if response.text:
                    cleaned_json = self.clean_gemini_response(response.text)
                    result = json.loads(cleaned_json)

                    # Validate the result structure
                    required_keys = ["key_parties", "critical_dates", "party_obligations", "red_flags"]
                    if all(key in result for key in required_keys):
                        return result
                    else:
                        logger.warning(f"Invalid result structure on attempt {attempt + 1}")
                        continue
                else:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    continue

            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                continue

        # Return empty result if all attempts failed
        logger.error("All analysis attempts failed")
        return {
            "key_parties": [],
            "critical_dates": [],
            "party_obligations": {},
            "red_flags": []
        }

    def merge_results(self, chunk_results):
        """
        Merge and deduplicate results from multiple chunks
        """
        final_results = {
            "key_parties": [],
            "critical_dates": [],
            "party_obligations": {},
            "red_flags": []
        }

        # Use sets for efficient deduplication
        seen_parties = set()
        seen_dates = set()
        seen_flags = set()

        for result in chunk_results:
            # Merge key_parties
            for party in result.get("key_parties", []):
                party_tuple = (party.get("name", ""), party.get("role", ""))
                if party_tuple[0] and party_tuple not in seen_parties:
                    final_results["key_parties"].append(party)
                    seen_parties.add(party_tuple)

            # Merge critical_dates
            for date in result.get("critical_dates", []):
                date_tuple = (date.get("event", ""), date.get("date", ""))
                if date_tuple[0] and date_tuple not in seen_dates:
                    final_results["critical_dates"].append(date)
                    seen_dates.add(date_tuple)

            # Merge red_flags
            for flag in result.get("red_flags", []):
                if flag and flag not in seen_flags:
                    final_results["red_flags"].append(flag)
                    seen_flags.add(flag)

            # Merge party_obligations
            for party_name, obligations in result.get("party_obligations", {}).items():
                if party_name not in final_results["party_obligations"]:
                    final_results["party_obligations"][party_name] = set()
                final_results["party_obligations"][party_name].update(obligations)

        # Convert sets back to lists
        for party_name, ob_set in final_results["party_obligations"].items():
            final_results["party_obligations"][party_name] = sorted(list(ob_set))

        logger.info("Results merged and deduplicated successfully")
        return final_results

    def process_document(self, full_document_text: str):
        """
        Main function to process a legal document
        """
        try:
            logger.info("Starting document analysis")

            # Step 1: Create chunks
            chunks = self.create_chunks(full_document_text)

            # Step 2: Analyze each chunk (Map phase)
            logger.info("Analyzing chunks...")
            chunk_results = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
                result = self.analyze_chunk(chunk)
                chunk_results.append(result)

            # Step 3: Merge results (Reduce phase)
            logger.info("Merging results...")
            final_results = self.merge_results(chunk_results)

            logger.info("Document analysis completed successfully")
            return final_results

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {
                "error": f"Failed to process document: {str(e)}",
                "key_parties": [],
                "critical_dates": [],
                "party_obligations": {},
                "red_flags": []
            }


# Global analyzer instance
analyzer = DocumentAnalyzer()


def process_document(full_document_text: str):
    """
    Convenience function for external use
    """
    return analyzer.process_document(full_document_text)


# Test function
if __name__ == "__main__":
    sample_text = """
    This Freelance Agreement ('Agreement'), effective October 1, 2025, is between Innovate Inc. ('Client') 
    and Jane Doe ('Freelancer'). Freelancer agrees to deliver the project by November 15, 2025. 
    This contract is governed by the laws of California. Innovate Inc. must pay the agreed upon fee 
    within 30 days of project completion. The agreement may be terminated by either party with 30 days notice.
    """

    result = process_document(sample_text)
    print(json.dumps(result, indent=2))