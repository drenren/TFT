"""TFT Interpretability Module."""

from .attention_weights import (
    extract_attention_weights,
    average_attention_over_heads,
    get_temporal_attention_patterns,
    analyze_attention_focus,
    compute_attention_entropy,
    plot_attention_heatmap,
    plot_attention_by_head,
)

from .variable_importance import (
    extract_variable_selection_weights,
    compute_average_variable_importance,
    rank_variables_by_importance,
    analyze_temporal_variable_importance,
    compute_variable_interactions,
    plot_variable_importance,
    plot_temporal_variable_importance,
    plot_variable_importance_heatmap,
    compare_encoder_decoder_importance,
)

__all__ = [
    # Attention
    'extract_attention_weights',
    'average_attention_over_heads',
    'get_temporal_attention_patterns',
    'analyze_attention_focus',
    'compute_attention_entropy',
    'plot_attention_heatmap',
    'plot_attention_by_head',
    # Variable importance
    'extract_variable_selection_weights',
    'compute_average_variable_importance',
    'rank_variables_by_importance',
    'analyze_temporal_variable_importance',
    'compute_variable_interactions',
    'plot_variable_importance',
    'plot_temporal_variable_importance',
    'plot_variable_importance_heatmap',
    'compare_encoder_decoder_importance',
]
