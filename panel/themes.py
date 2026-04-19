"""Theme management for the panel web UI."""
import json
from pathlib import Path
from typing import Optional

# Path to themes definition file
THEMES_PATH = Path(__file__).parent / 'themes.json'

# Default theme if none configured
DEFAULT_THEME_ID = 'forest-floor'

# Built-in fallback so the panel can still boot if themes.json is absent.
FALLBACK_THEMES = {
    'themes': [
        {
            'id': DEFAULT_THEME_ID,
            'name': 'Forest Floor',
            'type': 'dark',
            'colors': {
                'bg-base': '#07130F',
                'bg-card': '#0E1C17',
                'bg-subtle': '#163128',
                'text-primary': '#E7F2EC',
                'text-muted': '#9AB8AB',
                'accent': '#2FB073',
                'accent-hover': '#24915E',
                'success': '#4CBF84',
                'error': '#D16A5F',
                'warning': '#D7A64C',
                'border': '#22473A',
                'border-strong': '#356854',
            },
            'chart': [
                '#2FB073',
                '#5FCB97',
                '#7CC7B5',
                '#A5D66A',
                '#D7A64C',
                '#D16A5F',
                '#4D8F76',
                '#7FBF8E',
            ],
        }
    ]
}

# Cached themes data
_themes_cache: Optional[dict] = None


def load_themes() -> list[dict]:
    """Load all available themes from themes.json.
    
    Returns:
        List of theme dictionaries with id, name, type, colors, chart, etc.
    """
    global _themes_cache
    
    if _themes_cache is not None:
        return _themes_cache['themes']
    
    if not THEMES_PATH.exists():
        _themes_cache = FALLBACK_THEMES
        return _themes_cache['themes']
    
    with open(THEMES_PATH, encoding='utf-8') as f:
        data = json.load(f)
    
    _themes_cache = data
    return data['themes']


def get_theme(theme_id: str) -> dict:
    """Get a specific theme by ID.
    
    Args:
        theme_id: The theme identifier (e.g., 'morning-light', 'night-kitchen')
        
    Returns:
        Theme dictionary with all color definitions
        
    Raises:
        ValueError: If theme_id not found
    """
    themes = load_themes()
    
    for theme in themes:
        if theme['id'] == theme_id:
            return theme
    
    # Fall back to default if requested theme not found
    for theme in themes:
        if theme['id'] == DEFAULT_THEME_ID:
            return theme
    
    # Last resort: return first theme
    return themes[0]


def get_themes_json() -> str:
    """Get themes as JSON string for client-side use.
    
    Returns:
        JSON string containing array of theme objects
    """
    themes = load_themes()
    return json.dumps(themes)


def get_theme_choices() -> list[tuple[str, str, str]]:
    """Get themes formatted for a select dropdown.
    
    Returns:
        List of tuples: (id, name, type) for each theme
    """
    themes = load_themes()
    return [(t['id'], t['name'], t['type']) for t in themes]


def reload_themes():
    """Force reload themes from disk (useful after editing themes.json)."""
    global _themes_cache
    _themes_cache = None
    return load_themes()


def validate_theme_id(theme_id: str) -> str:
    """Validate and return a theme ID, falling back to default if invalid.
    
    Args:
        theme_id: Theme ID to validate
        
    Returns:
        Valid theme ID (original or default)
    """
    themes = load_themes()
    valid_ids = {t['id'] for t in themes}
    
    if theme_id in valid_ids:
        return theme_id
    
    return DEFAULT_THEME_ID
