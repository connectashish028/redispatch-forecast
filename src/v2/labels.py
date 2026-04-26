"""
labels.py - plain-English labels and short descriptions for UI display.

Internal feature/horizon names stay technical (matches column names in
features.parquet and the trained models). The dashboard / notebooks call
into this module to translate them for end users.

Usage
-----
    from labels import HORIZON_LABEL, FEATURE_LABEL, METRIC_LABEL, pretty
    HORIZON_LABEL['y_24h']        # 'Next 24 hours'
    FEATURE_LABEL['wind_onshore'] # 'Onshore wind generation (MW)'
    pretty('wx_wind_speed_100m')  # 'Wind speed at 100m altitude (m/s)'
"""
from __future__ import annotations


HORIZON_LABEL: dict[str, str] = {
    'y_1h':  'Next 1 hour',
    'y_6h':  'Next 6 hours',
    'y_24h': 'Next 24 hours',
}

HORIZON_DESCRIPTION: dict[str, str] = {
    'y_1h':  'Probability of any redispatch event happening in the hour right after now.',
    'y_6h':  'Probability of any redispatch event happening any time in the next 6 hours.',
    'y_24h': 'Probability of any redispatch event happening any time in the next 24 hours (tomorrow).',
}


METRIC_LABEL: dict[str, str] = {
    'pr_auc_cal':  'Ranking quality (PR-AUC)',
    'pr_auc_raw':  'Ranking quality before calibration',
    'pr_auc_lift': 'Lift vs random guess',
    'roc_auc':     'Discrimination (ROC-AUC)',
    'brier_cal':   'Probability error (Brier)',
    'brier_raw':   'Probability error before calibration',
    'log_loss_cal':'Log loss',
    'base_rate':   'How often the event happens',
    'best_iteration': 'Trees used',
}

METRIC_HELP: dict[str, str] = {
    'pr_auc_cal':  'How well the model ranks busy days above quiet days. '
                   '0 = random, 1 = perfect. Compare to "How often the event happens".',
    'roc_auc':     'How well the model separates positives from negatives at any threshold. '
                   '0.5 = random, 1.0 = perfect. Above 0.8 is good for forecasting.',
    'brier_cal':   'Squared error between predicted probability and what actually happened. '
                   'Lower is better. Below 0.10 is good for rare events.',
    'pr_auc_lift': 'Times better than predicting the average rate. 4x means the '
                   'model gets 4 events for every 1 a random guess would get at the same volume.',
    'base_rate':   'Fraction of (hour, town) rows where redispatch actually happened. '
                   'A trivial "always say yes" model has PR-AUC = base rate.',
    'best_iteration': 'How many decision trees the model needed before it stopped improving. '
                      'Tiny numbers mean the signal is concentrated in a few features.',
}


FEATURE_LABEL: dict[str, str] = {
    # target / status
    'is_active':              'Redispatch happening now',

    # lag / persistence
    'active_24h_lag24':       'Active hours in the day before forecast time',
    'active_7d_lag24':        'Active hours in the week before forecast time',

    # SMARD national
    'wind_onshore':           'Onshore wind generation (MW)',
    'wind_offshore':          'Offshore wind generation (MW)',
    'solar':                  'Solar generation (MW)',
    'total_load':             'Total electricity demand (MW)',
    'residual_load':          'Residual demand after renewables (MW)',
    'day_ahead':              'Day-ahead market price (EUR/MWh)',

    # weather (Berlin proxy, hourly)
    'wx_wind_speed_100m':     'Wind speed at 100m altitude (m/s)',
    'wx_wind_speed_10m':      'Wind speed at 10m altitude (m/s)',
    'wx_wind_direction_100m': 'Wind direction at 100m (degrees)',
    'wx_wind_direction_10m':  'Wind direction at 10m (degrees)',
    'wx_wind_gusts_10m':      'Wind gusts at 10m (m/s)',
    'wx_shortwave_radiation': 'Solar radiation reaching ground (W/m^2)',
    'wx_direct_radiation':    'Direct solar radiation (W/m^2)',
    'wx_diffuse_radiation':   'Scattered solar radiation (W/m^2)',
    'wx_temperature_2m':      'Air temperature at 2m (degC)',
    'wx_apparent_temperature':'Apparent temperature (degC)',
    'wx_cloud_cover':         'Cloud cover (%)',
    'wx_cloud_cover_low':     'Low cloud cover (%)',
    'wx_cloud_cover_mid':     'Mid cloud cover (%)',
    'wx_cloud_cover_high':    'High cloud cover (%)',
    'wx_precipitation':       'Precipitation (mm)',
    'wx_rain':                'Rain (mm)',
    'wx_snowfall':            'Snowfall (mm)',
    'wx_pressure_msl':        'Sea-level pressure (hPa)',
    'wx_relative_humidity_2m':'Humidity at 2m (%)',

    # geography
    'lat':                    'Latitude (north <-> south)',
    'lon':                    'Longitude (east <-> west)',

    # calendar
    'hour':                   'Hour of day (0-23)',
    'dow':                    'Day of week (Mon=0)',
    'month':                  'Month (1-12)',
    'is_weekend':             'Weekend?',
    'is_de_holiday':          'German public holiday?',

    # collective
    'town_*':                 'Town identity (175 columns combined)',
}


def pretty(name: str) -> str:
    """Return the plain-English label for any feature/horizon/metric name.
    Falls back to the raw name if not in the dictionaries."""
    if name in HORIZON_LABEL:
        return HORIZON_LABEL[name]
    if name in FEATURE_LABEL:
        return FEATURE_LABEL[name]
    if name in METRIC_LABEL:
        return METRIC_LABEL[name]
    if name.startswith('town_'):
        return f'Town: {name[5:]}'
    return name
