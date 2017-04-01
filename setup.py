from setuptools import setup

config = {
    'include_package_data': True,
    'description': 'Simulations of DNA',
    'download_url': 'https://github.com/kundajelab/simdna',
    'version': '0.3',
    'packages': ['simdna', 'simdna.resources', 'simdna.synthetic'],
    'package_data': {'simdna.resources': ['encode_motifs.txt.gz', 'HOCOMOCOv10_HUMAN_mono_homer_format_0.001.motif.gz']},
    'setup_requires': [],
    'install_requires': ['numpy>=1.9', 'matplotlib', 'scipy'],
    'dependency_links': [],
    'scripts': ['scripts/densityMotifSimulation.py',
                'scripts/emptyBackground.py',
                'scripts/motifGrammarSimulation.py'],
    'name': 'simdna'
}

if __name__== '__main__':
    setup(**config)
