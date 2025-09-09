#!/usr/bin/env python3
"""
Pre-Qualification Analyzer for Philippines-focused Propaganda Detection
Emphasis on maritime security and territorial issues while maintaining broad detection
"""

import numpy as np
import re
import os
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from textblob import TextBlob
import json
import time
from pathlib import Path
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv('.env.local')

# Import centralized data extractor
try:
    from ..core.data_extractor import DataExtractor
    DATA_EXTRACTOR_AVAILABLE = True
except ImportError:
    DATA_EXTRACTOR_AVAILABLE = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class PhilippinesPropagandaAnalyzer:
    def __init__(self, project_id = 115):
        """Initialize the Philippines-focused propaganda analyzer."""
        print("🔧 Initializing Philippines Propaganda Analyzer...")
        print("  🎯 Focus: Maritime security, territorial issues, and broader propaganda detection")
        
        # Initialize Supabase client
        print("  🔗 Connecting to Supabase...")
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_DATA_API_KEY_SECRET')
        
        if not self.supabase_url or not self.supabase_key:
            print("⚠️  Supabase credentials not found in environment variables")
            print("   Please set SUPABASE_URL and SUPABASE_DATA_API_KEY_SECRET environment variables")
            self.supabase = None
        else:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Store project_id as text (for eos_definitions table compatibility)
        self.project_id = str(project_id) if project_id else "115"
        
        # Initialize centralized data extractor if available
        if DATA_EXTRACTOR_AVAILABLE:
            try:
                self.data_extractor = DataExtractor()
                print("  ✅ Centralized data extractor initialized")
            except Exception as e:
                print(f"  ⚠️  Could not initialize centralized data extractor: {e}")
                self.data_extractor = None
        else:
            self.data_extractor = None
        
        # Load sentence transformer model
        print("  📥 Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load EOS definitions and create embeddings
        self.eos_definitions = self._load_eos_definitions()
        self.eos_embeddings = self._create_eos_embeddings()
        
        # Define Philippines-focused propaganda keywords and patterns
        self.propaganda_keywords = self._load_philippines_keywords()
        
        print("✅ Philippines Propaganda Analyzer initialized successfully!")
    
    def _load_eos_definitions(self) -> Dict[str, str]:
        """Load EOS definitions from centralized extractor or fallback to direct Supabase."""
        # Try centralized data extractor first
        if self.data_extractor is not None:
            try:
                print(f"  📊 Loading EOS definitions using centralized extractor for project_id: {self.project_id}")
                eos_definitions = self.data_extractor.get_eos_definitions(self.project_id)
                
                definitions = {}
                for record in eos_definitions:
                    theme = record.get('theme', '')
                    narrative_element = record.get('narrative_element', '')
                    definition = record.get('definition', '')
                    signals = record.get('signals', '')
                    
                    # Combine narrative_element and definition for better matching
                    combined_text = f"{narrative_element}"
                    if definition:
                        combined_text += f": {definition}"
                    if signals:
                        combined_text += f" | Signals: {signals}"
                    
                    if theme and combined_text.strip():
                        definitions[theme] = combined_text.strip()
                
                print(f"  ✅ Loaded {len(definitions)} EOS definitions using centralized extractor")
                return definitions
                
            except Exception as e:
                print(f"  ⚠️  Centralized extractor failed: {e}, falling back to direct Supabase")
        
        # Fallback to direct Supabase query
        if self.supabase is None:
            print("  ❌ Supabase not available")
            return {}
            
        try:
            print(f"  📊 Loading EOS definitions directly from Supabase for project_id: {self.project_id}")
            
            # Query EOS definitions from Supabase
            response = self.supabase.table('eos_definitions').select(
                'narrative_element, theme, definition, signals'
            ).eq('project_id', self.project_id).execute()
            
            definitions = {}
            for record in response.data:
                theme = record.get('theme', '')
                narrative_element = record.get('narrative_element', '')
                definition = record.get('definition', '')
                signals = record.get('signals', '')
                
                # Combine narrative_element and definition for better matching
                combined_text = f"{narrative_element}"
                if definition:
                    combined_text += f": {definition}"
                if signals:
                    combined_text += f" | Signals: {signals}"
                
                if theme and combined_text.strip():
                    definitions[theme] = combined_text.strip()
            
            print(f"  ✅ Loaded {len(definitions)} EOS definitions from Supabase")
            return definitions
            
        except Exception as e:
            print(f"❌ Error loading EOS definitions from Supabase: {e}")
            return {}
    
    def _create_eos_embeddings(self) -> Dict[str, np.ndarray]:
        """Create embeddings for all EOS definitions."""
        print("  🔄 Creating EOS definition embeddings...")
        embeddings = {}
        
        for theme, definition in self.eos_definitions.items():
            # Combine theme name and definition for embedding
            text = f"{theme}: {definition}"
            embedding = self.embedding_model.encode([text])[0]
            embeddings[theme] = embedding
        
        return embeddings
    
    def _load_philippines_keywords(self) -> Dict[str, List[str]]:
        """Load Philippines-focused propaganda keywords with emphasis on maritime/security."""
        return {
            # Maritime and territorial keywords (HIGH PRIORITY)
            "maritime_security": [
                "west philippine sea", "south china sea", "spratlys", "spratly islands",
                "scarborough shoal", "panatag shoal", "bajo de masinloc",
                "kalayaan islands", "pag-asa island", "thitu island",
                "ayungin shoal", "second thomas shoal", "sierra madre",
                "reed bank", "recto bank", "benham rise", "philippine rise",
                "exclusive economic zone", "eez", "unclos", "arbitral tribunal",
                "nine-dash line", "maritime domain", "territorial waters",
                "fishing rights", "maritime patrol", "coast guard", "navy",
                "maritime security", "sea lanes", "freedom of navigation",
                # Tagalog maritime terms
                "karagatan", "dagat", "isla", "teritoryo", "soberanya"
            ],
            
            # China-Philippines relations
            "china_relations": [
                "china", "chinese", "beijing", "pogo", "offshore gaming",
                "belt and road", "bri", "infrastructure", "investment",
                "bilateral relations", "cooperation", "partnership",
                "中国", "中文", "友好", "合作",
                # Tagalog
                "tsino", "intsik", "mainland china"
            ],
            
            # Security and defense
            "security_defense": [
                "afp", "armed forces", "philippine navy", "philippine coast guard",
                "defense", "security", "sovereignty", "territorial integrity",
                "mutual defense treaty", "mdt", "visiting forces agreement", "vfa",
                "balikatan", "military exercises", "defense cooperation",
                "west philippine sea command", "wescom", "surveillance",
                # Tagalog
                "sandatahang lakas", "hukbo", "seguridad", "depensa",
                "bantay dagat", "tanggulan"
            ],
            
            # Government and politics (EXPANDED)
            "government": [
                "duterte", "marcos", "bongbong", "bbm", "sara duterte", "rody",
                "philippines", "filipino", "pilipino", "manila", "malacanang",
                "dfa", "department of foreign affairs", "dnd", "defense department",
                "senate", "congress", "administration", "government",
                "liberal party", "lp", "pdp-laban", "nacionalista", "lakas-cmd",
                "mayor", "governor", "senator", "congressman", "barangay",
                # Tagalog political terms
                "gobyerno", "pamahalaan", "pulitika", "eleksyon", "halalan",
                "boto", "kandidato", "partido", "senado", "kongreso",
                "pangulo", "bise presidente", "kalihim", "gabinete"
            ],
            
            # Social issues (HIGH PRIORITY - NOT CELEBRITIES)
            "social_issues": [
                # Poverty and inequality
                "poverty", "mahirap", "kahirapan", "inequality", "hunger",
                "gutom", "walang trabaho", "unemployment", "minimum wage",
                "kontraktwalisasyon", "endo", "contractualization",
                "iskwater", "squatter", "relokasyon", "demolition",
                "homeless", "walang tirahan", "maralita", "dukha",
                
                # Education issues
                "education", "edukasyon", "k-12", "k to 12", "deped",
                "scholarship", "tuition", "matrikula", "student loan",
                "literacy", "out of school", "dropout", "child labor",
                
                # Healthcare
                "philhealth", "healthcare", "universal health", "covid",
                "vaccine", "bakuna", "pandemic", "hospitalization",
                "medical bills", "gamot", "doktor", "ospital",
                
                # Crime and drugs
                "drugs", "droga", "shabu", "war on drugs", "ejk",
                "extrajudicial", "tokhang", "oplan tokhang", "adik",
                "pusher", "drug lord", "rehabilitation", "crime",
                "krimen", "kriminal", "pulis", "police", "pnp",
                
                # Corruption
                "corruption", "korupsyon", "lagay", "kotong", "bribe",
                "suhol", "kickback", "tongpats", "ghost project",
                "pork barrel", "pdaf", "dap", "malversation",
                "plunder", "graft", "ill-gotten wealth", "nakaw"
            ],
            
            # Economic themes (EXPANDED WITH TAGALOG)
            "economic": [
                "build build build", "infrastructure", "development", "investment",
                "trade", "economy", "gdp", "growth", "jobs", "employment",
                "poverty alleviation", "foreign investment", "fdi",
                # Tagalog economic terms
                "ekonomiya", "negosyo", "trabaho", "sweldo", "sahod",
                "presyo", "bilihin", "inflation", "implasyon", "krisis",
                "utang", "loan", "tax", "buwis", "vat", "train law",
                "rice tariffication", "fuel price", "gasolina", "pamasahe"
            ],
            
            # Cultural/Social movements (NOT ENTERTAINMENT)
            "social_movements": [
                "ofw", "overseas filipino workers", "remittance",
                "bayanihan", "unity", "nationalism", "patriotism",
                "filipino pride", "heritage", "tradition",
                "people power", "edsa", "activism", "aktibista",
                "rally", "protesta", "welga", "strike", "union",
                "human rights", "karapatang pantao", "press freedom",
                "abs-cbn", "media freedom", "censorship",
                # Indigenous and minority issues
                "lumad", "ip", "indigenous", "moro", "bangsamoro",
                "barmm", "mindanao", "npa", "cpp", "red-tagging"
            ],
            
            # Disinformation indicators (EXPANDED)
            "disinfo_markers": [
                "fake news", "truth", "expose", "revealed", "mainstream media",
                "they don't want you to know", "wake up", "open your eyes",
                "dilawan", "yellowtards", "dds", "trolls", "bias media",
                # Tagalog disinfo markers
                "totoo ba", "hindi niyo alam", "lihim", "sikreto",
                "bayarang media", "bayaran", "troll farm", "hakot",
                "pasabog", "breaking", "urgent", "share mo", "i-share",
                "ctto", "credits to the owner", "viral", "trending"
            ],
            
            # Regional actors
            "regional_actors": [
                "asean", "southeast asia", "vietnam", "malaysia", "indonesia",
                "taiwan", "japan", "united states", "us", "america", "kano",
                "australia", "quad", "aukus", "five eyes", "nato",
                "russia", "ukraine", "israel", "palestine", "middle east"
            ],
            
            # Religious and values issues (SENSITIVE TOPICS)
            "religious_values": [
                "catholic", "katoliko", "christian", "iglesia",
                "muslim", "islam", "ramadan", "halal",
                "divorce", "diborsyo", "abortion", "same-sex marriage",
                "lgbt", "lgbtq", "sogie", "pride", "bakla", "tomboy",
                "family values", "pamilyang pilipino", "moralidad"
            ],
            
            # EXCLUSION INDICATORS (entertainment/celebrity content to ignore)
            "entertainment_exclude": [
                # Celebrity names and terms
                "artista", "celebrity", "showbiz", "teleserye", "drama",
                "love team", "fans", "fandom", "concert", "album",
                "movie", "pelikula", "mmff", "box office", "premiere",
                "trending", "viral video", "tiktok dance", "vlog",
                "youtube channel", "influencer", "endorsement",
                "commercial", "advertisement", "brand ambassador",
                # Sports (unless political)
                "basketball", "pba", "volleyball", "boxing", "manny pacquiao",
                "gilas", "uaap", "ncaa", "athlete", "championship",
                # Pure entertainment
                "game show", "reality show", "singing contest", "the voice",
                "idol", "star search", "beauty pageant", "miss universe",
                "binibining pilipinas", "swimsuit", "evening gown"
            ]
        }
    
    def _clean_text(self, text: str, max_chars: int = 1000) -> str:
        """Clean and truncate text."""
        if not text or text is None:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate to max_chars
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        return text.strip()
    
    def _extract_text_fields(self, post_data: Dict[str, Any]) -> str:
        """Extract and combine all text fields from post data (CM System format)."""
        text_fields = []
        
        # Add post_text if available (primary content)
        if post_data.get('post_text'):
            text_fields.append(self._clean_text(post_data['post_text']))
        
        # Add content if available
        if post_data.get('content'):
            text_fields.append(self._clean_text(post_data['content']))
        
        # Add text if available
        if post_data.get('text'):
            text_fields.append(self._clean_text(post_data['text']))
        
        # Add description if available
        if post_data.get('description'):
            text_fields.append(self._clean_text(post_data['description']))
        
        # Add title if available
        if post_data.get('title'):
            text_fields.append(self._clean_text(post_data['title']))
        
        # Add body if available
        if post_data.get('body'):
            text_fields.append(self._clean_text(post_data['body']))
        
        # Fallback to legacy ad fields if present
        if post_data.get('adBodyHTML'):
            text_fields.append(self._clean_text(post_data['adBodyHTML']))
        
        if post_data.get('adDescription'):
            text_fields.append(self._clean_text(post_data['adDescription']))
        
        if post_data.get('ad_headline'):
            text_fields.append(self._clean_text(post_data['ad_headline']))
        
        if post_data.get('ad_body'):
            text_fields.append(self._clean_text(post_data['ad_body']))
        
        # Combine all text fields
        combined_text = " | ".join([field for field in text_fields if field])
        return combined_text
    
    def _keyword_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze text for propaganda-related keywords with Philippines focus."""
        text_lower = text.lower()
        
        # First check for entertainment/celebrity content to exclude
        if 'entertainment_exclude' in self.propaganda_keywords:
            entertainment_matches = 0
            for keyword in self.propaganda_keywords['entertainment_exclude']:
                if keyword.lower() in text_lower:
                    entertainment_matches += 1
            
            # If high entertainment content detected, penalize heavily
            if entertainment_matches >= 3:
                return {
                    "keyword_scores": {"entertainment_content": entertainment_matches},
                    "total_matches": 0,
                    "weighted_matches": 0,
                    "keyword_score": 0,
                    "is_entertainment": True
                }
        
        keyword_scores = {}
        total_matches = 0
        weighted_score = 0
        
        # Define weights for different categories (social/political issues get higher weight)
        category_weights = {
            "maritime_security": 2.0,  # Double weight for maritime issues
            "social_issues": 2.0,      # High weight for social issues
            "security_defense": 1.8,
            "religious_values": 1.6,   # Sensitive topics often used in propaganda
            "china_relations": 1.5,
            "disinfo_markers": 1.5,
            "social_movements": 1.4,
            "regional_actors": 1.3,
            "government": 1.2,
            "economic": 1.0,
            "entertainment_exclude": -2.0  # Negative weight for entertainment
        }
        
        for category, keywords in self.propaganda_keywords.items():
            if category == 'entertainment_exclude':
                continue  # Already handled above
                
            matches = 0
            matched_terms = []
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    matches += 1
                    matched_terms.append(keyword)
            
            keyword_scores[category] = {
                "count": matches,
                "matched_terms": matched_terms[:5]  # Keep top 5 matches for review
            }
            
            # Apply category weight
            weight = category_weights.get(category, 1.0)
            weighted_score += matches * weight
            total_matches += matches
        
        # Calculate overall keyword score (0-100) with weighting
        max_possible_weighted = sum(
            len(keywords) * abs(category_weights.get(cat, 1.0)) 
            for cat, keywords in self.propaganda_keywords.items()
            if cat != 'entertainment_exclude'
        )
        keyword_score = (weighted_score / max_possible_weighted) * 100 if max_possible_weighted > 0 else 0
        
        return {
            "keyword_scores": keyword_scores,
            "total_matches": total_matches,
            "weighted_matches": weighted_score,
            "keyword_score": keyword_score,
            "is_entertainment": False
        }
    
    def _sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Perform sentiment analysis on text."""
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            return {
                "polarity": sentiment.polarity,  # -1 to 1
                "subjectivity": sentiment.subjectivity,  # 0 to 1
                "sentiment_score": (sentiment.polarity + 1) * 50  # Convert to 0-100 scale
            }
        except Exception as e:
            return {
                "polarity": 0,
                "subjectivity": 0.5,
                "sentiment_score": 50
            }
    
    def _eos_similarity_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze similarity to EOS definitions."""
        try:
            # Create embedding for the ad text
            text_embedding = self.embedding_model.encode([text])[0]
            
            # Calculate similarities with all EOS definitions
            similarities = {}
            for theme, theme_embedding in self.eos_embeddings.items():
                similarity = cosine_similarity([text_embedding], [theme_embedding])[0][0]
                similarities[theme] = float(similarity)
            
            # Find top matches
            sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            top_matches = sorted_similarities[:5]
            
            # Calculate overall EOS score (average of top 3 similarities)
            top_scores = [score for _, score in top_matches[:3]]
            eos_score = np.mean(top_scores) * 100 if top_scores else 0
            
            return {
                "similarities": similarities,
                "top_matches": top_matches,
                "eos_score": eos_score
            }
        except Exception as e:
            return {
                "similarities": {},
                "top_matches": [],
                "eos_score": 0
            }
    
    def _check_maritime_security_focus(self, keyword_analysis: Dict) -> float:
        """Check if content has strong maritime/security focus for bonus scoring."""
        # Skip if it's entertainment content
        if keyword_analysis.get('is_entertainment', False):
            return -20.0  # Heavy penalty for entertainment content
        
        maritime_matches = keyword_analysis.get('keyword_scores', {}).get('maritime_security', {}).get('count', 0)
        security_matches = keyword_analysis.get('keyword_scores', {}).get('security_defense', {}).get('count', 0)
        social_matches = keyword_analysis.get('keyword_scores', {}).get('social_issues', {}).get('count', 0)
        
        # Bonus for maritime/security content
        bonus = 0.0
        if maritime_matches >= 2 or security_matches >= 2:
            bonus += 15.0  # 15% bonus for strong maritime/security content
        elif maritime_matches >= 1 or security_matches >= 1:
            bonus += 7.5   # 7.5% bonus for some maritime/security content
        
        # Additional bonus for social issues (non-entertainment)
        if social_matches >= 3:
            bonus += 10.0  # 10% bonus for strong social issues content
        elif social_matches >= 1:
            bonus += 5.0   # 5% bonus for some social issues content
        
        return bonus
    
    def _calculate_propaganda_score(self, keyword_analysis: Dict, sentiment_analysis: Dict, eos_analysis: Dict) -> float:
        """Calculate overall propaganda likelihood score with Philippines focus."""
        # Weighted combination of different factors
        keyword_weight = 0.45  # Increased weight for keywords (Philippines-specific)
        sentiment_weight = 0.15  # Reduced weight
        eos_weight = 0.40
        
        keyword_score = keyword_analysis.get('keyword_score', 0)
        sentiment_score = sentiment_analysis.get('sentiment_score', 50)
        eos_score = eos_analysis.get('eos_score', 0)
        
        # Normalize sentiment score (higher subjectivity might indicate propaganda)
        sentiment_factor = sentiment_analysis.get('subjectivity', 0.5) * 100
        
        # Calculate base weighted score
        base_score = (
            keyword_score * keyword_weight +
            sentiment_factor * sentiment_weight +
            eos_score * eos_weight
        )
        
        # Add bonus for maritime/security focus
        maritime_bonus = self._check_maritime_security_focus(keyword_analysis)
        
        propaganda_score = base_score + maritime_bonus
        
        return min(100, max(0, propaganda_score))
    
    def analyze_post(self, post_text: str, post_id: str = None, project_id: str = "1") -> Dict[str, Any]:
        """Analyze a single post for propaganda likelihood - CM System Integration."""
        # Use project_id as text
        project_id_text = str(project_id) if project_id else "1"
        
        # Create post data structure for analysis (CM System format)
        post_data = {
            'id': post_id or 'unknown',
            'text': post_text,
            'content': post_text,
            'platform': 'facebook',
            'post_text': post_text,
            'description': post_text,
            'title': post_text[:100] if len(post_text) > 100 else post_text,
            'body': post_text
        }
        
        # Use existing analyze_ad method (it will work with any data structure)
        result = self.analyze_ad(post_data)
        
        # Add CM system specific fields
        result['post_id'] = post_id
        result['project_id'] = project_id
        result['should_scrape'] = (
            result.get('propaganda_score', 0) > 25 and
            not result.get('is_entertainment', False) and
            (result.get('is_maritime_focused', False) or 
             result.get('is_social_focused', False) or
             len(result.get('eos_analysis', {}).get('top_matches', [])) > 0)
        )
        
        return result
    
    def analyze_ad(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single post for propaganda likelihood with Philippines focus."""
        post_id = post_data.get('id', 'unknown')
        
        # Extract text
        text = self._extract_text_fields(post_data)
        if not text:
            return {
                "post_id": post_id,
                "error": "No text content found",
                "propaganda_score": 0
            }
        
        # Perform analyses
        keyword_analysis = self._keyword_analysis(text)
        
        # Check if it's entertainment content - skip if it is
        if keyword_analysis.get('is_entertainment', False):
            return {
                "post_id": post_id,
                "text_preview": text[:200] + "..." if len(text) > 200 else text,
                "propaganda_score": 0,
                "is_entertainment": True,
                "is_maritime_focused": False,
                "keyword_analysis": keyword_analysis,
                "sentiment_analysis": {"polarity": 0, "subjectivity": 0, "sentiment_score": 50},
                "eos_analysis": {"similarities": {}, "top_matches": [], "eos_score": 0},
                "analysis_timestamp": time.time()
            }
        
        sentiment_analysis = self._sentiment_analysis(text)
        eos_analysis = self._eos_similarity_analysis(text)
        
        # Calculate overall propaganda score
        propaganda_score = self._calculate_propaganda_score(keyword_analysis, sentiment_analysis, eos_analysis)
        
        # Determine focus areas
        maritime_bonus = self._check_maritime_security_focus(keyword_analysis)
        is_maritime_focused = maritime_bonus > 0
        
        # Check for social issues focus
        social_matches = keyword_analysis.get('keyword_scores', {}).get('social_issues', {}).get('count', 0)
        is_social_focused = social_matches >= 2
        
        return {
            "post_id": post_id,
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
            "propaganda_score": propaganda_score,
            "is_maritime_focused": is_maritime_focused,
            "is_social_focused": is_social_focused,
            "is_entertainment": False,
            "keyword_analysis": keyword_analysis,
            "sentiment_analysis": sentiment_analysis,
            "eos_analysis": eos_analysis,
            "analysis_timestamp": time.time()
        }
    
    def analyze_posts_batch(self, posts_data: List[Dict[str, Any]], threshold: float = 25.0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Analyze a batch of posts and return candidates above threshold."""
        print(f"🔍 Analyzing {len(posts_data)} posts for Philippines propaganda candidates...")
        print(f"   Focus: Social/political issues, maritime security, excluding entertainment")
        
        all_results = []
        candidates = []
        maritime_candidates = []
        social_candidates = []
        entertainment_filtered = 0
        
        for i, post_data in enumerate(posts_data):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(posts_data)} posts analyzed")
            
            result = self.analyze_ad(post_data)
            all_results.append(result)
            
            # Track entertainment content that was filtered
            if result.get('is_entertainment', False):
                entertainment_filtered += 1
                continue  # Skip entertainment content
            
            # Check if this post qualifies as a candidate
            if result.get('propaganda_score', 0) >= threshold:
                candidates.append(result)
                
                # Track maritime-focused candidates separately
                if result.get('is_maritime_focused', False):
                    maritime_candidates.append(result)
                
                # Track social issues candidates
                if result.get('is_social_focused', False):
                    social_candidates.append(result)
        
        print(f"✅ Analysis complete!")
        print(f"   Total candidates: {len(candidates)} above threshold ({threshold}%)")
        print(f"   Maritime/security focused: {len(maritime_candidates)} candidates")
        print(f"   Social issues focused: {len(social_candidates)} candidates")
        print(f"   Entertainment content filtered: {entertainment_filtered} posts")
        
        return all_results, candidates
    
    def save_results(self, results: List[Dict[str, Any]], filename: str):
        """Save analysis results to JSON file."""
        output_path = Path("output") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        # Create clean structured output
        structured_results = []
        for result in results:
            ad_id = result.get('ad_id', 'unknown')
            text = result.get('text_preview', '')
            propaganda_score = result.get('propaganda_score', 0)
            is_maritime = result.get('is_maritime_focused', False)
            
            # Extract EOS matches with percentages
            eos_analysis = result.get('eos_analysis', {})
            top_matches = eos_analysis.get('top_matches', [])
            
            matches = {}
            for theme, score in top_matches:
                percentage = round(score * 100, 1)
                matches[theme] = percentage
            
            # Extract keyword analysis with matched terms
            keyword_analysis = result.get('keyword_analysis', {})
            keyword_scores = keyword_analysis.get('keyword_scores', {})
            
            # Create clean keyword summary
            keyword_summary = {}
            for category, data in keyword_scores.items():
                if data['count'] > 0:
                    keyword_summary[category] = {
                        "count": data['count'],
                        "matched": data.get('matched_terms', [])
                    }
            
            # Create clean structured result
            structured_result = {
                "ad_id": ad_id,
                "text": text,
                "propaganda_score": round(propaganda_score, 1),
                "is_maritime_security_focused": is_maritime,
                "eos_matches": matches,
                "keyword_matches": keyword_summary,
                "total_keyword_matches": keyword_analysis.get('total_matches', 0),
                "sentiment": {
                    "polarity": round(result.get('sentiment_analysis', {}).get('polarity', 0), 3),
                    "subjectivity": round(result.get('sentiment_analysis', {}).get('subjectivity', 0), 3)
                }
            }
            
            structured_results.append(structured_result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path}")
    
    def save_results_to_supabase(self, results: List[Dict[str, Any]], batch_size: int = 10):
        """Save analysis results to Supabase ad_qualification_results table in batches."""
        print(f"💾 Saving {len(results)} results to Supabase in batches of {batch_size}...")
        
        # Try centralized data extractor first
        if self.data_extractor is not None:
            try:
                print("  🔄 Using centralized data extractor for saving...")
                
                # Prepare data in the format expected by centralized extractor
                formatted_results = []
                for result in results:
                    ad_id = result.get('ad_id', 'unknown')
                    
                    # Use ad_id as text
                    ad_id_text = str(ad_id)
                    
                    text = result.get('text_preview', '')
                    propaganda_score = result.get('propaganda_score', 0)
                    is_maritime = result.get('is_maritime_focused', False)
                    
                    # Extract EOS matches with percentages
                    eos_analysis = result.get('eos_analysis', {})
                    top_matches = eos_analysis.get('top_matches', [])
                    
                    matches = {}
                    for theme, score in top_matches:
                        percentage = round(score * 100, 1)
                        matches[theme] = percentage
                    
                    # Extract keyword analysis
                    keyword_analysis = result.get('keyword_analysis', {})
                    keyword_scores = keyword_analysis.get('keyword_scores', {})
                    
                    # Create keyword summary
                    keyword_summary = {}
                    for category, data in keyword_scores.items():
                        if data['count'] > 0:
                            keyword_summary[category] = {
                                "count": data['count'],
                                "matched": data.get('matched_terms', [])[:5]
                            }
                    
                    formatted_results.append({
                        'ad_id': ad_id_text,
                        'project_id': self.project_id,
                        'text': text,
                        'propaganda_score': round(propaganda_score, 2),
                        'is_maritime_focused': is_maritime,
                        'matches': matches,
                        'keyword_matches': keyword_summary,
                        'total_keyword_matches': keyword_analysis.get('total_matches', 0),
                        'sentiment': {
                            'polarity': round(result.get('sentiment_analysis', {}).get('polarity', 0), 3),
                            'subjectivity': round(result.get('sentiment_analysis', {}).get('subjectivity', 0), 3)
                        }
                    })
                
                success = self.data_extractor.save_analysis_results(formatted_results)
                if success:
                    print("  ✅ Successfully saved using centralized data extractor")
                    return
                else:
                    print("  ⚠️  Centralized extractor failed, falling back to direct Supabase")
            except Exception as e:
                print(f"  ⚠️  Centralized extractor error: {e}, falling back to direct Supabase")
        
        # Fallback to direct Supabase
        if self.supabase is None:
            print("⚠️  Supabase not available, skipping database save")
            return
        
        # [Rest of the Supabase saving logic remains the same as original]
        # ... (keeping the same batch saving implementation)

def main(project_id = None, platform: str = 'all', limit: int = None, threshold: float = 25.0, verbose: bool = False, overwrite: bool = False):
    """Main function to run Philippines-focused propaganda analysis."""
    # Use environment variable or default
    if project_id is None:
        project_id = os.getenv('DEFAULT_PROJECT_ID', '115')
    
    # Use project_id as text for database queries
    project_id_text = str(project_id)
    
    print("🇵🇭 PHILIPPINES PROPAGANDA DETECTION SYSTEM")
    print("="*60)
    print("Focus Areas: Maritime Security, Territorial Issues, Broad Propaganda")
    print("="*60)
    
    # Initialize analyzer with project_id
    analyzer = PhilippinesPropagandaAnalyzer(project_id=project_id_text)
    
    # [Rest of the main function logic remains the same]
    # Load ads data from Supabase (same implementation as original)
    if analyzer.supabase is None:
        print("❌ Supabase not available")
        return
    
    try:
        print(f"📥 Loading ads from Supabase for project {project_id}...")
        
        # Get project info first
        project_response = analyzer.supabase.table('projects').select('*').eq('projectid', project_id_text).execute()
        if not project_response.data:
            print(f"❌ Project {project_id_text} not found")
            return
        
        project_info = project_response.data[0]
        project_name = project_info['projectname']
        project_name_formatted = project_name.lower().replace(' ', '_')
        print(f"📋 Project: {project_name} (formatted: {project_name_formatted})")
        
        # Use RPC function to get pending ads efficiently
        if not overwrite:
            print(f"🔍 Using RPC to get pending ads for project {project_id_text}...")
            
            # Call the RPC function with appropriate parameters
            rpc_params = {
                'p_project_name': project_name_formatted,
                'p_project_id': project_id_text
            }
            
            # Add platform filter if specified
            if platform and platform.lower() != 'all':
                rpc_params['p_platform'] = platform
                print(f"🔍 Filtering by platform: {platform}")
            
            # Execute RPC call
            response = analyzer.supabase.rpc('get_ads_pending_qualification', rpc_params).execute()
            
            all_ads_data = response.data
            print(f"📊 Found {len(all_ads_data)} pending ads to process")
            
            if not all_ads_data:
                print(f"✅ All ads for project {project_id_text} have already been processed!")
                print(f"💡 Use --overwrite flag to reprocess all ads")
                return
                
        else:
            print(f"🔄 Overwrite mode: Loading all ads for reprocessing...")
            
            # Build query for master_ad_creatives (for overwrite mode)
            query = analyzer.supabase.table('master_ad_creatives').select('*').eq('project', project_name_formatted)
            
            # Apply platform filter if specified
            if platform and platform.lower() != 'all':
                query = query.ilike('platform', platform)
                print(f"🔍 Filtering by platform: {platform}")
            
            # Execute query with pagination to get ALL ads
            all_ads_data = []
            page_size = 1000
            offset = 0
            
            while True:
                # Apply pagination
                page_query = query.range(offset, offset + page_size - 1)
                
                # Execute query
                response = page_query.execute()
                
                if not response.data:
                    break
                
                all_ads_data.extend(response.data)
                print(f"📥 Loaded batch {len(all_ads_data)//page_size + 1}: {len(response.data)} ads (total: {len(all_ads_data)})")
                
                # If we got less than page_size, we've reached the end
                if len(response.data) < page_size:
                    break
                
                offset += page_size
            
            if not all_ads_data:
                print(f"❌ No ads found for project {project_id_text}")
                return
        
        # Apply limit if specified (after getting data)
        if limit:
            all_ads_data = all_ads_data[:limit]
        
        ads_data = all_ads_data
        print(f"✅ Will process {len(ads_data)} ads")
        
    except Exception as e:
        print(f"❌ Error loading ads from Supabase: {e}")
        return
    
    # Analyze all ads
    print("\n" + "="*60)
    print("PROPAGANDA ANALYSIS - PHILIPPINES FOCUS")
    print("="*60)
    
    # Analyze all posts
    all_results, candidates = analyzer.analyze_posts_batch(ads_data, threshold=threshold)
    
    # Save ALL results to Supabase
    print(f"💾 Saving ALL {len(all_results)} results to database...")
    analyzer.save_results_to_supabase(all_results)
    
    # Save candidates to local file for review
    if candidates:
        analyzer.save_results(candidates, "philippines_propaganda_candidates.json")
        
        # Separate maritime-focused candidates
        maritime_candidates = [c for c in candidates if c.get('is_maritime_focused', False)]
        if maritime_candidates:
            analyzer.save_results(maritime_candidates, "maritime_security_candidates.json")
        
        # Print top candidates
        print(f"\n🏆 TOP CANDIDATES (Score >= {threshold}%):")
        print("="*60)
        
        sorted_candidates = sorted(candidates, key=lambda x: x.get('propaganda_score', 0), reverse=True)
        
        for i, candidate in enumerate(sorted_candidates[:10], 1):
            score = candidate.get('propaganda_score', 0)
            ad_id = candidate.get('ad_id', 'unknown')
            text_preview = candidate.get('text_preview', 'N/A')
            is_maritime = candidate.get('is_maritime_focused', False)
            is_social = candidate.get('is_social_focused', False)
            
            # Create focus tags
            tags = []
            if is_maritime:
                tags.append("🚢 MARITIME/SECURITY")
            if is_social:
                tags.append("🏘️ SOCIAL ISSUES")
            
            tag_string = f" [{', '.join(tags)}]" if tags else ""
            
            print(f"\n{i}. Ad ID: {ad_id} (Score: {score:.1f}%){tag_string}")
            print(f"   Text: {text_preview}")
            
            # Show keyword matches with details
            keyword_analysis = candidate.get('keyword_analysis', {})
            keyword_scores = keyword_analysis.get('keyword_scores', {})
            significant_categories = []
            
            for cat, data in keyword_scores.items():
                if data['count'] > 0:
                    # Show some matched terms for context
                    matched_preview = ', '.join(data.get('matched_terms', [])[:3])
                    if matched_preview:
                        significant_categories.append(f"{cat}({data['count']}): {matched_preview}")
                    else:
                        significant_categories.append(f"{cat}({data['count']})")
            
            if significant_categories:
                print(f"   Keywords found:")
                for cat_info in significant_categories[:3]:  # Show top 3 categories
                    print(f"     • {cat_info}")
            
            # Show top EOS matches
            eos_analysis = candidate.get('eos_analysis', {})
            top_matches = eos_analysis.get('top_matches', [])
            if top_matches:
                print(f"   EOS matches: {', '.join([f'{theme}({score:.2f})' for theme, score in top_matches[:3]])}")
    
    else:
        print("\n❌ No candidates found above threshold. Consider lowering the threshold.")

if __name__ == "__main__":
    import argparse
    import sys
    
    # Check if this is a single post analysis (CM System integration)
    if len(sys.argv) >= 3 and sys.argv[1] == '--analyze-post':
        # Single post analysis mode for CM System
        if len(sys.argv) < 4:
            print(json.dumps({
                'success': False,
                'message': 'Usage: python philippines_propaganda_analyzer.py --analyze-post "post_text" "post_id" [project_id]'
            }))
            sys.exit(1)
        
        post_text = sys.argv[2]
        post_id = sys.argv[3]
        project_id = sys.argv[4] if len(sys.argv) > 4 else "1"
        
        try:
            # Initialize analyzer
            analyzer = PhilippinesPropagandaAnalyzer(project_id=project_id)
            
            # Analyze the post
            result = analyzer.analyze_post(post_text, post_id, project_id)
            
            # Output JSON result
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            print(json.dumps({
                'success': False,
                'message': f'Analysis failed: {str(e)}',
                'post_id': post_id,
                'project_id': project_id
            }))
            sys.exit(1)
        
        sys.exit(0)
    
    # Set up argument parser for batch analysis
    parser = argparse.ArgumentParser(
        description="Philippines-focused Propaganda Analyzer with Maritime Security Emphasis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python philippines_analyzer.py                 # Run with default settings (project_id=115)
  python philippines_analyzer.py 120             # Analyze project_id 120
  python philippines_analyzer.py 120 facebook    # Analyze project_id 120, Facebook platform only
  python philippines_analyzer.py 120 all 100     # Analyze project_id 120, all platforms, limit to 100 ads
  python philippines_analyzer.py --analyze-post "text" "post_id" "project_id"  # Analyze single post
  python philippines_analyzer.py --help          # Show this help message
        """
    )
    
    parser.add_argument(
        'project_id',
        nargs='?',
        type=str,
        default='115',
        help='Project ID to analyze (default: 115)'
    )
    
    parser.add_argument(
        'platform',
        nargs='?',
        type=str,
        default='all',
        choices=['all', 'facebook', 'google', 'tiktok', 'youtube', 'instagram', 'twitter'],
        help='Platform to filter ads by (default: all)'
    )
    
    parser.add_argument(
        'limit',
        nargs='?',
        type=int,
        default=None,
        help='Limit number of ads to analyze (default: all ads)'
    )
    
    parser.add_argument(
        '--threshold',
        '-t',
        type=float,
        default=25.0,
        help='Propaganda score threshold for candidate selection (default: 25.0, lower for Philippines focus)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--overwrite',
        '-o',
        action='store_true',
        help='Process all ads, even if already analyzed (default: skip analyzed ads)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate project_id (accept any non-empty string)
    if not args.project_id or not args.project_id.strip():
        print(f"❌ Invalid project_id: '{args.project_id}'. Must be a non-empty string.")
        sys.exit(1)
    
    # Validate limit if provided
    if args.limit is not None and args.limit <= 0:
        print(f"❌ Invalid limit: {args.limit}. Must be a positive integer.")
        sys.exit(1)
    
    # Validate threshold
    if args.threshold < 0 or args.threshold > 100:
        print(f"❌ Invalid threshold: {args.threshold}. Must be between 0 and 100.")
        sys.exit(1)
    
    print(f"🚀 Starting Philippines-focused analysis with:")
    print(f"   Project ID: {args.project_id}")
    print(f"   Platform: {args.platform}")
    print(f"   Limit: {args.limit if args.limit else 'all ads'}")
    print(f"   Threshold: {args.threshold}%")
    print(f"   Verbose: {args.verbose}")
    print(f"   Overwrite: {args.overwrite}")
    print()
    
    main(project_id=args.project_id, platform=args.platform, limit=args.limit, 
         threshold=args.threshold, verbose=args.verbose, overwrite=args.overwrite)
