"""
Neuroscience Reproducibility Analysis using OpenAlex API
Modified for specific search criteria:
- Title/Abstract: brain OR neuroscience
- Years: 2015-2025
- Type: article
- Open Access only
"""

import requests
import json
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt


class OpenAlexNeuroscienceReview:
    """
    A class to query OpenAlex for neuroscience articles with reproducibility focus
    and analyze their code/data sharing practices.
    """
    
    def __init__(self, email: str, output_dir: str = "./neuroscience_repro_output"):
        """
        Initialize the OpenAlex client.
        
        Parameters:
        -----------
        email : str
            Your email for polite pool access (faster API)
        output_dir : str
            Directory to save outputs
        """
        self.base_url = "https://api.openalex.org/works"
        self.email = email
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Polite pool header for faster access
        self.headers = {
            'User-Agent': f'NeuroReproducibilityReview/1.0 (mailto:{email})'
        }

    #query to take a number of subject randomly for the year range  
    # def build_custom_query(self) -> Dict:
    #     """
    #     Proper OpenAlex query:
    #     Filter = domain + metadata
    #     Search = topic keywords
    #     """

    #     filters = [
    #         "publication_year:2015-2025",
    #         "type:article",
    #         "is_oa:true",                # Open Access only
    #         "concepts.id:C41008148",   # Neuroscience ONLY
    #         'title_and_abstract.search:(brain  OR neuroscience )', #OR  neural
    #         #'title_and_abstract.search:(reproducible OR reproducibility OR "data sharing" OR "code sharing" OR github OR repository OR "data availability" OR "code availability" OR "supplementary data" OR "shared dataset")'

    #        # brain OR neural OR neuron OR neuroscience OR neurobiolog* OR neurophysiolog* OR neuroimaging OR fMRI OR EEG OR MEG OR PET OR cortex OR hippocampus OR synapse OR "nervous system"
    #        #Reproducible OR reproducibility OR "data sharing" OR "code sharing" OR "open data" OR "open code" OR "open science" OR "data availability" OR "data availability statement" OR "code availability" OR github OR gitlab OR repository OR "supplementary data" OR "shared dataset"


    #     ]

    #     filter_string = ",".join(filters)

    #     query_params = {
    #         'filter': filter_string,
    #         'per-page': 200,
    #         'mailto': self.email
    #     }

    #     return query_params
    
    def build_query_for_year(self, year: int) -> Dict:
        
        filters = [
            f"publication_year:{year}",
            "type:article",
            "is_oa:true", # Open Access only
            "concepts.id:C41008148",  # OpenAlex concept: Neuroscience
            'title_and_abstract.search:(brain OR neuroscience)' #edit here
        ]

        filter_string = ",".join(filters)

        query_params = {
            'filter': filter_string,
            'per-page': 200,
            'mailto': self.email
        }

        return query_params
    
    def search_articles(self, 
                        query_params: Dict,
                        max_results: int = 15000) -> List[Dict]:
        """
        Search OpenAlex for articles matching the query.
        
        Parameters:
        -----------
        query_params : Dict
            Query parameters for OpenAlex API
        max_results : int
            Maximum number of results to retrieve (default 15000)
            
        Returns:
        --------
        List of article dictionaries
        """
        all_results = []
        page = 1
        
        print(f"Starting OpenAlex search...")
        print(f"Filter: {query_params['filter']}")
    
        
        while len(all_results) < max_results:
            query_params['page'] = page
            
            try:
                response = requests.get(
                    self.base_url,
                    params=query_params,
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    if not results:
                        print(f"\nNo more results found. Total: {len(all_results)}")
                        break
                    
                    all_results.extend(results)
                    
                    # # Get metadata info
                    meta = data.get('meta', {})
                    total_count = meta.get('count', 0)
                    
                    print(f"Page {page}: Retrieved {len(results)} articles | "
                          f"Total so far: {len(all_results)}/{total_count}")

                    # print(f"Page {page}: Retrieved {len(results)} articles | Total so far: {len(all_results)}")
                    
                    # Check if we've retrieved all available results
                    # if not results:
    
                    if len(all_results) >= total_count:
                        print(f"\nRetrieved all {total_count} available articles!")
                        break
                    
                    page += 1
                    time.sleep(0.15)  
            
                elif response.status_code == 429:
                    # Rate limited - wait longer
                    print(f"Rate limited. Waiting 60 seconds...")
                    time.sleep(60)
                    
                else:
                    print(f"Error: {response.status_code} - {response.text}")
                    break
                    
            except requests.exceptions.Timeout:
                print(f"Timeout on page {page}. Retrying...")
                time.sleep(5)
                continue
                
            except Exception as e:
                print(f"Exception occurred: {e}")
                break
        
        print(f"\n{'='*60}")
        print(f"Search complete. Retrieved {len(all_results)} articles.")
        print(f"{'='*60}\n")
        
        return all_results[:max_results]
    
    def _reconstruct_abstract(self, inverted_index: Dict) -> str:
        """
        Reconstruct abstract text from OpenAlex inverted index.
        
        Parameters:
        -----------
        inverted_index : Dict
            Inverted index from OpenAlex
            
        Returns:
        --------
        Reconstructed abstract text
        """
        if not inverted_index:
            return ""
        
        # Create word-position pairs
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        
        # Sort by position and join
        word_positions.sort()
        return ' '.join([word for _, word in word_positions])
    
    def extract_metadata(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Extract relevant metadata from articles.
        
        Parameters:
        -----------
        articles : List[Dict]
            List of article dictionaries
            
        Returns:
        --------
        DataFrame with extracted metadata
        """
        print("Extracting metadata from articles...")
        records = []
        
        for i, article in enumerate(articles):
            if (i + 1) % 500 == 0:
                print(f"  Processing article {i + 1}/{len(articles)}...")
                
            record = {
                'id': article.get('id', ''),
                'openalex_id': article.get('id', '').split('/')[-1] if article.get('id') else '',
                'doi': article.get('doi', ''),
                'title': article.get('title', ''),
                'publication_year': article.get('publication_year'),
                'publication_date': article.get('publication_date'),
                'venue': self._get_venue_name(article),
                'citation_count': article.get('cited_by_count', 0),
                'type': article.get('type', ''),
                'is_oa': article.get('open_access', {}).get('is_oa', False),
                'oa_status': article.get('open_access', {}).get('oa_status', ''),
                'oa_url': article.get('open_access', {}).get('oa_url', ''),
                'abstract': self._reconstruct_abstract(
                    article.get('abstract_inverted_index', {})
                ),
                'concepts': ', '.join([c.get('display_name', '') 
                                      for c in article.get('concepts', [])[:5]]),
                'authors': self._get_authors(article),
                'institutions': self._get_institutions(article),
                'referenced_works_count': article.get('referenced_works_count', 0),
                'is_retracted': article.get('is_retracted', False),
                'is_paratext': article.get('is_paratext', False)
            }
            records.append(record)
        
        print(f"Metadata extraction complete: {len(records)} articles")
        return pd.DataFrame(records)
    
    def _get_venue_name(self, article: Dict) -> str:
        """Extract venue/journal name."""
        primary_location = article.get('primary_location', {})
        if primary_location:
            source = primary_location.get('source', {})
            if source:
                return source.get('display_name', 'Unknown')
        return 'Unknown'
    
    def _get_authors(self, article: Dict) -> str:
        """Extract author names."""
        authorships = article.get('authorships', [])
        authors = [a.get('author', {}).get('display_name', '') 
                  for a in authorships[:5]]  # First 5 authors
        return '; '.join([a for a in authors if a])
    
    def _get_institutions(self, article: Dict) -> str:
        """Extract institution names."""
        authorships = article.get('authorships', [])
        institutions = set()
        for authorship in authorships[:3]:  # First 3 authors
            for inst in authorship.get('institutions', []):
                inst_name = inst.get('display_name', '')
                if inst_name:
                    institutions.add(inst_name)
        return '; '.join(list(institutions)[:3])
    
    def check_code_data_sharing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for evidence of code/data sharing in titles and abstracts.
        
        This analysis distinguishes between:
        1. Code only
        2. Data only  
        3. Both code and data
        4. Neither
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with article metadata
            
        Returns:
        --------
        DataFrame with additional columns for code/data availability
        """
        print("\nAnalyzing code/data sharing patterns...")
        df['title'] = df['title'].fillna('')
        df['abstract'] = df['abstract'].fillna('')
        df['title'] = df['title'].astype(str)
        df['abstract'] = df['abstract'].astype(str)
        # ==============================================================    
        # STEP 1: Identify CODE-specific platforms and keywords
        # ==============================================================
        
        #parameters can be changed for your research
        code_platforms = {
            'github': r'github\.com|github\.io|github repository',
            'gitlab': r'gitlab\.com',
            'bitbucket': r'bitbucket\.org',
            'code_ocean': r'codeocean\.com',
            'jupyter': r'jupyter|jupyterlab|jupyterhub|notebook',
            'binder': r'mybinder\.org|binder',
            'docker': r'docker|dockerhub|docker hub'
        }
        
        # Check each CODE platform
        for platform, pattern in code_platforms.items():
            df[f'has_{platform}'] = (
                df['title'].str.contains(pattern, case=False, na=False, regex=True) |
                df['abstract'].str.contains(pattern, case=False, na=False, regex=True)
            )
        
        # CODE-specific keywords
        code_keywords = [
            r'\bcode\s+(available|shared|provided|repository|accessible|deposited)',
            r'source\s+code',
            r'code\s+is\s+available',
            r'reproducible\s+code',
            r'analysis\s+code',
            r'scripts?\s+(available|shared|provided)',
            r'software\s+(available|shared|provided)',
            r'implementation\s+(available|shared)',
            r'code\s+can\s+be\s+found',
            r'available\s+on\s+github',
            r'repository\s+contains.*code'
        ]
        

        df['has_code_keywords'] = df.apply(
            lambda row: any(
                pd.Series([row['title'] + ' ' + (row['abstract'] if pd.notna(row['abstract']) else '')])
                .str.contains(kw, case=False, na=False, regex=True).iloc[0] 
                for kw in code_keywords
            ), axis=1
        )
        
        # ==============================================================
        # STEP 2: Identify DATA-specific platforms and keywords
        # ==============================================================
        
        data_platforms = {
            'zenodo': r'zenodo\.org',
            'osf': r'osf\.io|open science framework',
            'figshare': r'figshare\.com',
            'dryad': r'datadryad\.org|dryad digital',
            'dataverse': r'dataverse'
        }
        
        # Check each DATA platform
        for platform, pattern in data_platforms.items():
            df[f'has_{platform}'] = (
                df['title'].str.contains(pattern, case=False, na=False, regex=True) |
                df['abstract'].str.contains(pattern, case=False, na=False, regex=True)
            )
        
        # DATA-specific keywords
        data_keywords = [
            r'\bdata\s+(available|shared|provided|repository|accessible|deposited)',
            r'data\s+is\s+available',
            r'dataset\s+(available|shared|provided)',
            r'raw\s+data',
            r'data\s+can\s+be\s+found',
            r'data\s+accessibility',
            r'publicly\s+available\s+data',
            r'data\s+repository',
            r"\bdata sharing\b",
            r'supplementary\s+data',
            r'open\s+data',
            r"\bdata are available\b",
        ]
        
        df['has_data_keywords'] = df.apply(
            lambda row: any(
                pd.Series([row['title'] + ' ' + (row['abstract'] if pd.notna(row['abstract']) else '')])
                .str.contains(kw, case=False, na=False, regex=True).iloc[0] 
                for kw in data_keywords
            ), axis=1
        )
        
        # ==============================================================
        # STEP 3: Determine if article shares CODE, DATA, or BOTH
        # ==============================================================
        
        # Code platforms columns
        code_platform_cols = [f'has_{p}' for p in code_platforms.keys()]
        
        # Data platforms columns  
        data_platform_cols = [f'has_{p}' for p in data_platforms.keys()]
        
        # Determine if shares CODE
        df['shares_code'] = (
            df[code_platform_cols].any(axis=1) | df['has_code_keywords']
        )
        
        # Determine if shares DATA
        df['shares_data'] = (
            df[data_platform_cols].any(axis=1) | df['has_data_keywords']
        )
        
        # Determine combined category
        df['shares_code_or_data'] = df['shares_code'] | df['shares_data']
        
        # Create explicit category variable (4 categories)
        df['sharing_category'] = 'neither'
        df.loc[df['shares_code'] & ~df['shares_data'], 'sharing_category'] = 'code_only'
        df.loc[~df['shares_code'] & df['shares_data'], 'sharing_category'] = 'data_only'
        df.loc[df['shares_code'] & df['shares_data'], 'sharing_category'] = 'code_and_data'
        
        # ==============================================================
        # STEP 4: Print summary statistics
        # ==============================================================
        
        print(f"\n  {'='*60}")
        print(f"  SHARING CATEGORIES:")
        print(f"  {'='*60}")
        
        category_counts = df['sharing_category'].value_counts()
        
        for category in ['code_and_data', 'code_only', 'data_only', 'neither']:
            count = category_counts.get(category, 0)
            pct = (count / len(df) * 100) if len(df) > 0 else 0
            category_label = category.replace('_', ' ').title()
            print(f"  {category_label:20s}: {count:4d} ({pct:5.1f}%)")
        
        total_sharing = df['shares_code_or_data'].sum()
        pct_total = (total_sharing / len(df) * 100) if len(df) > 0 else 0
        print(f"  {'-'*60}")
        print(f"  {'Total Sharing':20s}: {total_sharing:4d} ({pct_total:5.1f}%)")
        
        # Platform breakdown
        print(f"\n  {'='*60}")
        print(f"  CODE PLATFORMS:")
        print(f"  {'='*60}")
        for col in code_platform_cols:
            platform = col.replace('has_', '').replace('_', ' ').title()
            count = df[col].sum()
            if count > 0:
                print(f"  {platform:20s}: {count:4d}")
        if df['has_code_keywords'].sum() > 0:
            print(f"  {'Code Keywords':20s}: {df['has_code_keywords'].sum():4d}")
        
        print(f"\n  {'='*60}")
        print(f"  DATA PLATFORMS:")
        print(f"  {'='*60}")
        for col in data_platform_cols:
            platform = col.replace('has_', '').replace('_', ' ').title()
            count = df[col].sum()
            if count > 0:
                print(f"  {platform:20s}: {count:4d}")
        if df['has_data_keywords'].sum() > 0:
            print(f"  {'Data Keywords':20s}: {df['has_data_keywords'].sum():4d}")
        
        return df
    
    # # ============================================================
    # FIGURES
    # ============================================================
    def create_figures(self, df):
    
        from matplotlib import rcParams
        
        print("Creating figures...\n")
        
        # ═════════════════════════════════════════════════════════════════════════
        # Configuration globale du style scientifique
        # ═════════════════════════════════════════════════════════════════════════
        
        # Palette ColorBrewer qualitative (safe pour daltonisme + N&B)
        colors = {
            'code_and_data':  '#1b9e77',  # Teal foncé
            'code_only':      '#d95f02',  # Orange brûlé
            'data_only':      '#7570b3',  # Mauve
            'neither':        '#e7298a',  # Rose fuchsia
            'total':          '#333333',  # Gris très foncé
        }
        
        # Police et tailles
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        rcParams['font.size'] = 9
        rcParams['axes.labelsize'] = 10
        rcParams['axes.titlesize'] = 11
        rcParams['xtick.labelsize'] = 9
        rcParams['ytick.labelsize'] = 9
        rcParams['legend.fontsize'] = 9
        rcParams['figure.titlesize'] = 11
        
        # Grilles et axes
        rcParams['axes.linewidth'] = 0.8
        rcParams['grid.linewidth'] = 0.5
        rcParams['grid.alpha'] = 0.25
        rcParams['axes.edgecolor'] = '#333333'
        rcParams['axes.grid'] = False  # Grille manuelle pour plus de contrôle
        
        # ═════════════════════════════════════════════════════════════════════════
        # FIGURE 1 : Stacked Bar Chart (Nombre d'articles par année et catégorie)
        # ═════════════════════════════════════════════════════════════════════════
        
        # Preparing data 
        #df = df[df['sharing_category'] != 'neither'] #if you want to just show sharing articles in the bar graph 
        
        yearly_counts = pd.crosstab(
             df['publication_year'], 
             df['sharing_category'],
             margins=False
         )
        category_order = ['code_and_data', 'code_only', 'data_only','neither']
        #category_order = ['code_and_data', 'code_only', 'data_only'] #if you want to just show sharing articles in the bar graph 
        
        yearly_counts = yearly_counts[category_order]
        
        # Golden ratio : 6.4 × 4 pouces ≈ 16 × 10 cm (standard colonne simple)
        fig1, ax1 = plt.subplots(figsize=(6.4, 4.0), dpi=300)
        
        # Barres empilées avec bordures fines
        yearly_counts.plot(
            kind='bar', 
            stacked=True, 
            ax=ax1,
            color=[colors['code_and_data'], colors['code_only'], 
                colors['data_only'], colors['neither']],
            edgecolor='white',
            linewidth=0.4,
            width=0.75
        )
        
        # Labels et titre sobre
        ax1.set_xlabel('Publication year', fontweight='normal')
        ax1.set_ylabel('Number of articles', fontweight='normal')
        ax1.set_title('Evolution of data sharing practices (2015–2025)', 
                    fontweight='semibold', pad=12)
        
        # Légende repositionnée et étiquettes claires
        ax1.legend(
            title='Sharing category',
            labels=['Code + Data', 'Code only', 'Data only', 'Neither'], 
            #labels=['Code + Data', 'Code only', 'Data only'], #if you want to just show sharing articles in the bar graph 
            loc='upper left',
            frameon=True,
            edgecolor='#cccccc',
            framealpha=0.95,
            title_fontsize=9
        )
        
        # Rotation légère des labels x
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
        
        # Grille horizontale uniquement
        ax1.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#999999', alpha=0.4)
        ax1.set_axisbelow(True)
        
        # Limites y avec marge supérieure
        ax1.set_ylim(0, yearly_counts.sum(axis=1).max() * 1.08)
        
        # Retirer les épines supérieure et droite (style Tufte)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/fig1_sharing_bars.png", 
                    dpi=300, bbox_inches='tight', facecolor='white')
        # plt.savefig(f"{self.output_dir}/fig1_sharing_bars.pdf", 
        #             bbox_inches='tight', facecolor='white')  # Vectoriel pour LaTeX
        plt.close()
        
        # ═════════════════════════════════════════════════════════════════════════
        # FIGURE 2 : Line Chart (Taux de reproductibilité en %)
        # ═════════════════════════════════════════════════════════════════════════
        
        # Calculer les taux
        yearly_rates = df.groupby('publication_year').agg({
            'shares_code': 'mean',
            'shares_data': 'mean',
            'shares_code_or_data': 'mean'
        }).round(4) * 100
        
        yearly_rates['code_and_data'] = (
            df.groupby('publication_year')
            .apply(lambda x: (x['shares_code'] & x['shares_data']).mean() * 100)
        )
        
        # Golden ratio : 6.4 × 4 pouces
        fig2, ax2 = plt.subplots(figsize=(6.4, 4.0), dpi=300)
        
        years = yearly_rates.index
        
        # Lignes avec marqueurs distincts (lisibles en N&B)
        ax2.plot(years, yearly_rates['code_and_data'], 
                marker='o', linewidth=1.5, markersize=5, 
                color=colors['code_and_data'], 
                label='Code + Data',
                markerfacecolor='white', markeredgewidth=1.2)
        
        ax2.plot(years, yearly_rates['shares_code'], 
                marker='s', linewidth=1.5, markersize=5, 
                color=colors['code_only'], 
                label='Code only',
                markerfacecolor='white', markeredgewidth=1.2)
        
        ax2.plot(years, yearly_rates['shares_data'], 
                marker='^', linewidth=1.5, markersize=6, 
                color=colors['data_only'], 
                label='Data only',
                markerfacecolor='white', markeredgewidth=1.2)
        
        ax2.plot(years, yearly_rates['shares_code_or_data'], 
                marker='D', linewidth=2.0, markersize=5.5, 
                color=colors['total'], 
                linestyle='--', 
                label='Any sharing',
                markerfacecolor='white', markeredgewidth=1.2,
                zorder=5)  # Au premier plan
        
        # Labels
        ax2.set_xlabel('Publication year', fontweight='normal')
        ax2.set_ylabel('Sharing rate (%)', fontweight='normal')
        ax2.set_title('Trends in reproducibility rates (2015–2025)', 
                    fontweight='semibold', pad=12)
        
        # Légende compacte
        ax2.legend(loc='upper left', frameon=True, edgecolor='#cccccc', 
                framealpha=0.95, ncol=1)
        
        # Grille légère
        ax2.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#999999', alpha=0.4)
        ax2.xaxis.grid(True, linestyle=':', linewidth=0.5, color='#999999', alpha=0.25)
        ax2.set_axisbelow(True)
        
        # Limites et ticks
        ax2.set_ylim(0, min(100, yearly_rates.max().max() * 1.12))
        ax2.set_xlim(years.min() - 0.3, years.max() + 0.3)
        ax2.set_xticks(years)
        ax2.set_xticklabels(years, rotation=0)
        
        # Style Tufte
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/fig2_sharing_rates.png", 
                    dpi=300, bbox_inches='tight', facecolor='white')
        # plt.savefig(f"{self.output_dir}/fig2_sharing_rates.pdf", 
        #             bbox_inches='tight', facecolor='white')
        plt.close()
  
    def save_results(self, df: pd.DataFrame, filename: str = None):
        """
        Save results to CSV and JSON with summary statistics.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Results dataframe
        filename : str
            Base filename (without extension)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"neuroscience_reproducibility_{timestamp}"
        
        print(f"\nSaving results...")
        
        # Save CSV
        csv_path = os.path.join(self.output_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  CSV saved: {csv_path}")
        
        # Save JSON with full data
        json_path = os.path.join(self.output_dir, f"{filename}.json")
        df.to_json(json_path, orient='records', indent=2)
        print(f"  JSON saved: {json_path}")
        
        # Save detailed summary statistics
        summary_path = os.path.join(self.output_dir, f"{filename}_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("NEUROSCIENCE REPRODUCIBILITY ANALYSIS - DETAILED SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write("SEARCH CRITERIA:\n")
            f.write("-" * 50 + "\n")
            f.write("Title/Abstract: brain OR stroke OR neurotransmitter OR data OR\n")
            f.write("                code OR github OR neuroscience OR reproducible\n")
            f.write("Years: 2015-2025\n")
            f.write("Type: article\n")
            f.write("Open Access: yes\n\n")
            
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total articles retrieved: {len(df)}\n")
            f.write(f"Articles sharing code/data: {df['shares_code_or_data'].sum()}\n")
            f.write(f"Reproducibility rate: {df['shares_code_or_data'].mean()*100:.1f}%\n\n")
            
            f.write("SHARING CATEGORIES:\n")
            f.write("-" * 50 + "\n")
            category_counts = df['sharing_category'].value_counts()
            for category in ['code_and_data', 'code_only', 'data_only', 'neither']:
                count = category_counts.get(category, 0)
                pct = (count / len(df) * 100) if len(df) > 0 else 0
                category_label = category.replace('_', ' ').title()
                f.write(f"  {category_label:20s}: {count:4d} ({pct:5.1f}%)\n")
            
            f.write("\n  Summary:\n")
            f.write(f"  Articles sharing CODE: {df['shares_code'].sum()} ({df['shares_code'].mean()*100:.1f}%)\n")
            f.write(f"  Articles sharing DATA: {df['shares_data'].sum()} ({df['shares_data'].mean()*100:.1f}%)\n\n")
            
            f.write("CODE PLATFORMS/KEYWORDS:\n")
            f.write("-" * 50 + "\n")
            code_platforms = ['github', 'gitlab', 'bitbucket', 'code_ocean', 'jupyter', 'binder', 'docker']
            for platform in code_platforms:
                col = f'has_{platform}'
                if col in df.columns:
                    count = df[col].sum()
                    pct = (count / len(df) * 100) if len(df) > 0 else 0
                    if count > 0:
                        platform_label = platform.replace('_', ' ').title()
                        f.write(f"  {platform_label:20s}: {count:4d} ({pct:5.2f}%)\n")
            if 'has_code_keywords' in df.columns and df['has_code_keywords'].sum() > 0:
                count = df['has_code_keywords'].sum()
                pct = (count / len(df) * 100) if len(df) > 0 else 0
                f.write(f"  {'Code Keywords':20s}: {count:4d} ({pct:5.2f}%)\n")
            
            f.write("\nDATA PLATFORMS/KEYWORDS:\n")
            f.write("-" * 50 + "\n")
            data_platforms = ['zenodo', 'osf', 'figshare', 'dryad', 'dataverse']
            for platform in data_platforms:
                col = f'has_{platform}'
                if col in df.columns:
                    count = df[col].sum()
                    pct = (count / len(df) * 100) if len(df) > 0 else 0
                    if count > 0:
                        platform_label = platform.replace('_', ' ').title()
                        f.write(f"  {platform_label:20s}: {count:4d} ({pct:5.2f}%)\n")
            if 'has_data_keywords' in df.columns and df['has_data_keywords'].sum() > 0:
                count = df['has_data_keywords'].sum()
                pct = (count / len(df) * 100) if len(df) > 0 else 0
                f.write(f"  {'Data Keywords':20s}: {count:4d} ({pct:5.2f}%)\n")
            
            f.write("\nTEMPORAL ANALYSIS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Year range: {df['publication_year'].min()} - {df['publication_year'].max()}\n")
            
            # Reproducibility by year
            yearly_stats = df.groupby('publication_year').agg({
                'shares_code_or_data': ['sum', 'count', 'mean']
            }).round(3)
            f.write("\nReproducibility by year:\n")
            for year, row in yearly_stats.iterrows():
                sharing = int(row[('shares_code_or_data', 'sum')])
                total = int(row[('shares_code_or_data', 'count')])
                rate = row[('shares_code_or_data', 'mean')] * 100
                f.write(f"  {year}: {sharing}/{total} articles ({rate:.1f}%)\n")
            
            f.write("\nCITATION ANALYSIS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total citations: {df['citation_count'].sum()}\n")
            f.write(f"Average citations per article: {df['citation_count'].mean():.1f}\n")
            f.write(f"Median citations: {df['citation_count'].median():.0f}\n")
            
            # Compare citations for sharing vs non-sharing
            sharing = df[df['shares_code_or_data'] == True]
            not_sharing = df[df['shares_code_or_data'] == False]
            
            if len(sharing) > 0 and len(not_sharing) > 0:
                f.write(f"\nAvg citations (shares code/data): {sharing['citation_count'].mean():.1f}\n")
                f.write(f"Avg citations (no code/data): {not_sharing['citation_count'].mean():.1f}\n")
            
            # Citations by category
            f.write(f"\nCitations by sharing category:\n")
            for category in ['code_and_data', 'code_only', 'data_only', 'neither']:
                cat_df = df[df['sharing_category'] == category]
                if len(cat_df) > 0:
                    avg_cit = cat_df['citation_count'].mean()
                    category_label = category.replace('_', ' ').title()
                    f.write(f"  {category_label:20s}: {avg_cit:6.1f} avg citations\n")
            
            f.write(f"\nMost cited article:\n")
            most_cited = df.loc[df['citation_count'].idxmax()]
            f.write(f"  Title: {most_cited['title']}\n")
            f.write(f"  Citations: {most_cited['citation_count']}\n")
            f.write(f"  Year: {most_cited['publication_year']}\n")
            f.write(f"  Shares code/data: {most_cited['shares_code_or_data']}\n")
            
            f.write("\nTOP VENUES:\n")
            f.write("-" * 50 + "\n")
            top_venues = df['venue'].value_counts().head(10)
            for venue, count in top_venues.items():
                f.write(f"  {venue}: {count} articles\n")
        
        print(f"  Summary saved: {summary_path}")
        print(f"\nAll results saved to: {self.output_dir}/")


def main():
    """
    Main execution function for the analysis.
    """
    print("="*70)
    print("NEUROSCIENCE REPRODUCIBILITY ANALYSIS")
    print("="*70)
    print("\nSearch Criteria:")
    print("  Title/Abstract: brain OR neuroscience ")
    print("  Years: 2015-2025")
    print("  Type: article")
    print("  Open Access: yes")
    print("="*70 + "\n")
    
    # Initialize the review tool
    reviewer = OpenAlexNeuroscienceReview(
        email="your.email@example.com",
        output_dir="./neuroscience_repro_output"
    )
    
    #build query for each year one by one
    all_articles = []

    for year in range(2015, 2026):
        print(f"\n===== YEAR {year} =====")
        print("Step 1: Building query...")
        query_params = reviewer.build_query_for_year(year)
        print("\nStep 2: Searching OpenAlex...")
        articles = reviewer.search_articles(
            query_params=query_params,
            max_results=1000  # safe limite par année
        )
        
        print(f"Retrieved {len(articles)} articles for {year}")
        
        all_articles.extend(articles)

    print(f"\nTOTAL ARTICLES: {len(all_articles)}")
    # Remove duplicates based on OpenAlex ID
    unique_articles = {article['id']: article for article in all_articles}
    all_articles = list(unique_articles.values())

    print(f"After deduplication: {len(all_articles)} articles")
    # Build custom query
    # print("Step 1: Building query...")
    # query_params = reviewer.build_custom_query()
    
    # # Search for articles
    # print("\nStep 2: Searching OpenAlex...")
    # articles = reviewer.search_articles(
    #     query_params=query_params,
    #     max_results=10000  # OpenAlex limit
    # )
    
    # Extract metadata
    # print("\nStep 3: Extracting metadata...")
    # df = reviewer.extract_metadata(articles)
    df = reviewer.extract_metadata(all_articles)
    
    # Analyze code/data sharing
    print("\nStep 4: Analyzing code/data sharing...")
    df = reviewer.check_code_data_sharing(df)

    #Create figures
    reviewer.create_figures(df)
    
    # Save results
    print("\nStep 5: Saving results...")
    reviewer.save_results(df, filename="neuroscience_reproducibility_full")
    
    # Print final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Total articles analyzed: {len(df)}")
    print(f"Articles sharing code/data: {df['shares_code_or_data'].sum()}")
    print(f"Reproducibility rate: {df['shares_code_or_data'].mean()*100:.1f}%")
    print(f"\nYear range: {df['publication_year'].min()}-{df['publication_year'].max()}")
    print(f"Total citations: {df['citation_count'].sum()}")
    print("="*70 + "\n")
    
    return df


if __name__ == "__main__":
    results_df = main()
