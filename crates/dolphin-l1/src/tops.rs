//! TOPS azimuth carrier and the Sentinel-1 timing corrections.
//!
//! **Carrier.** The TOPS antenna sweep leaves each burst with an azimuth
//! chirp: the Doppler centroid moves linearly through the burst at rate
//! `k_t(τ) = k_s k_a / (k_a − k_s)`, `k_s = 2 v_s ψ̇ / λ`. Its phase
//! `φ(η, τ) = π k_t(τ) (η − η_ref(τ))²` (η from the burst centre line,
//! `η_ref(τ) = f_dc(τ₀)/k_a(τ₀) − f_dc(τ)/k_a(τ)`) is removed before any
//! azimuth interpolation and restored afterwards (ESA, *Sentinel-1 TOPS SLC
//! Deramping*, S1-TN-MDA-52-7445; s1-reader `az_carrier_components`).
//!
//! **Timing corrections** (Gisinger et al. 2021, TGRS 59(2); ETAD ATBD,
//! ETAD-DLR-DD-0008), the set OPERA CSLC-S1 applies to geolocation:
//! - bistatic azimuth delay `τ_mid/2 + τ/2 − rank·PRI` (IPF removed only the
//!   bulk term at the IW2 mid-swath);
//! - azimuth FM-rate mismatch: the annotated `k_a` against the one the orbit
//!   and target geometry imply;
//! - Doppler-induced range shift: (geometric + steering Doppler) / chirp rate;
//! - static troposphere: 2.3 m zenith delay, scale height 6 km, mapped by
//!   `1/cos(incidence)`.
//!
//! Azimuth corrections are subtracted from the zero-Doppler time and range
//! corrections added to the slant range before the SLC is sampled, matching
//! COMPASS's `az_lut = −(bistatic + fm)` / `rg_lut = doppler + tropo`.
//! Ionosphere and solid-earth tides are date-varying phase screens handled by
//! `dolphin-corrections` downstream, so they are not applied here.

use crate::error::{L1Error, Result};
use crate::geometry::{dot, norm, sub};
use crate::orbit::{Orbit, PlatformState, Vec3};
use crate::safe::{RangePolynomial, S1Burst, SPEED_OF_LIGHT};

/// Static troposphere zenith delay (m).
const STATIC_ZENITH_DELAY: f64 = 2.3;
/// Static troposphere scale height (m).
const STATIC_SCALE_HEIGHT: f64 = 6_000.0;

/// The TOPS azimuth carrier of one burst.
#[derive(Debug, Clone)]
pub struct TopsCarrier {
    steering_rate_hz: f64,
    azimuth_time_interval: f64,
    centre_line: f64,
    starting_range: f64,
    eta_ref_origin: f64,
    doppler: RangePolynomial,
    fm_rate: RangePolynomial,
}

impl TopsCarrier {
    /// The carrier of `burst`, with the platform speed at the burst middle.
    pub fn new(burst: &S1Burst, orbit: &Orbit) -> Result<Self> {
        let mid = orbit.seconds(burst.sensing_start)
            + 0.5 * burst.lines as f64 * burst.azimuth_time_interval;
        let speed = norm(orbit.state(mid)?.velocity);
        let doppler = burst.doppler()?.clone();
        let fm_rate = burst.fm_rate()?.clone();
        let r0 = burst.starting_range();
        Ok(Self {
            steering_rate_hz: 2.0 * speed * burst.azimuth_steering_rate / burst.wavelength(),
            azimuth_time_interval: burst.azimuth_time_interval,
            centre_line: (burst.lines / 2) as f64,
            starting_range: r0,
            eta_ref_origin: doppler.eval_range(r0) / fm_rate.eval_range(r0),
            doppler,
            fm_rate,
        })
    }

    /// Range-dependent terms `(k_t, η_ref)` at slant range `range`.
    pub fn range_terms(&self, range: f64) -> (f64, f64) {
        let ka = self.fm_rate.eval_range(range);
        let kt = self.steering_rate_hz / (1.0 - self.steering_rate_hz / ka);
        let eta_ref = self.eta_ref_origin - self.doppler.eval_range(range) / ka;
        (kt, eta_ref)
    }

    /// η (s from the burst centre line) of fractional burst line `line`.
    pub fn eta(&self, line: f64) -> f64 {
        (line - self.centre_line) * self.azimuth_time_interval
    }

    /// Carrier phase (rad) at fractional `line` and slant range `range`.
    pub fn phase(&self, line: f64, range: f64) -> f64 {
        let (kt, eta_ref) = self.range_terms(range);
        let d = self.eta(line) - eta_ref;
        std::f64::consts::PI * kt * d * d
    }

    /// Steering Doppler (Hz) at fractional `line` and slant range `range`.
    pub fn steering_doppler(&self, line: f64, range: f64) -> f64 {
        let (kt, eta_ref) = self.range_terms(range);
        kt * (self.eta(line) - eta_ref)
    }

    /// Total Doppler (steering + geometric, Hz).
    pub fn total_doppler(&self, line: f64, range: f64) -> f64 {
        self.steering_doppler(line, range) + self.doppler.eval_range(range)
    }

    /// Slant range of the first burst sample (m).
    pub fn starting_range(&self) -> f64 {
        self.starting_range
    }

    /// Baseband phase (rad) at fractional `line` and slant range `range`:
    /// `2π f_dc(R) η`. The deramped burst is centred on the geometric
    /// Doppler centroid `f_dc`; removing it before the sinc interpolation and
    /// restoring it after (ISCE3 `geocode::baseband`) keeps the spectrum in
    /// the kernel's passband. With `|f_dc|` of 60+ Hz, skipping this costs
    /// ~1 % amplitude and 0.999 correlation against OPERA.
    pub fn baseband_phase(&self, line: f64, range: f64) -> f64 {
        std::f64::consts::TAU * self.doppler.eval_range(range) * self.eta(line)
    }
}

/// Polynomial coefficients interpolated linearly along azimuth time.
#[derive(Debug, Clone)]
struct PolynomialTrack {
    times: Vec<f64>,
    t0: Vec<f64>,
    coefficients: Vec<[f64; 3]>,
}

impl PolynomialTrack {
    fn new(list: &[RangePolynomial], orbit: &Orbit) -> Result<Self> {
        if list.len() < 2 {
            return Err(L1Error::Product(
                "fewer than two Doppler/FM-rate estimates span the burst".into(),
            ));
        }
        let coefficient =
            |p: &RangePolynomial, i: usize| p.coefficients.get(i).copied().unwrap_or(0.0);
        Ok(Self {
            times: list.iter().map(|p| orbit.seconds(p.azimuth_time)).collect(),
            t0: list.iter().map(|p| p.t0).collect(),
            coefficients: list
                .iter()
                .map(|p| [coefficient(p, 0), coefficient(p, 1), coefficient(p, 2)])
                .collect(),
        })
    }

    /// Evaluate at azimuth time `t` and two-way range time `tau`: τ₀ is held
    /// at the end values outside the span; coefficients extrapolate linearly.
    fn eval(&self, t: f64, tau: f64) -> f64 {
        let last = self.times.len() - 1;
        let i = self.times.partition_point(|&x| x <= t).clamp(1, last) - 1;
        let w = (t - self.times[i]) / (self.times[i + 1] - self.times[i]);
        let t0 = self.t0[i] + w.clamp(0.0, 1.0) * (self.t0[i + 1] - self.t0[i]);
        let c: [f64; 3] = std::array::from_fn(|k| {
            self.coefficients[i][k] + w * (self.coefficients[i + 1][k] - self.coefficients[i][k])
        });
        let x = tau - t0;
        c[0] + x * (c[1] + x * c[2])
    }
}

/// The timing corrections of one burst.
#[derive(Debug, Clone)]
pub struct TimingCorrections {
    carrier: TopsCarrier,
    radar_frequency: f64,
    steering_rate_rad: f64,
    bistatic_offset: f64,
    range_chirp_rate: f64,
    first_line_time: f64,
    azimuth_time_interval: f64,
    fm_track: PolynomialTrack,
    dc_track: PolynomialTrack,
    burst_mid_time: f64,
    burst_mid_doppler_over_fm: f64,
}

/// Azimuth and range corrections at one target.
#[derive(Debug, Clone, Copy)]
pub struct Correction {
    /// Added to the zero-Doppler azimuth time (s).
    pub azimuth_time: f64,
    /// Added to the geometric slant range (m).
    pub slant_range: f64,
}

impl TimingCorrections {
    /// Corrections for `burst` on `orbit`.
    pub fn new(burst: &S1Burst, orbit: &Orbit) -> Result<Self> {
        let carrier = TopsCarrier::new(burst, orbit)?;
        let fm_track = PolynomialTrack::new(burst.spanning(&burst.azimuth_fm_rates), orbit)?;
        let dc_track = PolynomialTrack::new(burst.spanning(&burst.doppler_centroids), orbit)?;
        let first_line_time = orbit.seconds(burst.sensing_start);
        let burst_mid_time =
            first_line_time + 0.5 * (burst.lines - 1) as f64 * burst.azimuth_time_interval;
        let mid_tau = 2.0
            * (burst.starting_range() + 0.5 * burst.samples as f64 * burst.range_pixel_spacing())
            / SPEED_OF_LIGHT;
        let pri = 1.0 / burst.prf_raw;
        Ok(Self {
            radar_frequency: burst.radar_frequency,
            steering_rate_rad: burst.azimuth_steering_rate,
            bistatic_offset: burst.iw2_mid_range / SPEED_OF_LIGHT - f64::from(burst.rank) * pri,
            range_chirp_rate: burst.range_chirp_rate,
            first_line_time,
            azimuth_time_interval: burst.azimuth_time_interval,
            burst_mid_doppler_over_fm: dc_track.eval(burst_mid_time, mid_tau)
                / fm_track.eval(burst_mid_time, mid_tau),
            fm_track,
            dc_track,
            burst_mid_time,
            carrier,
        })
    }

    /// The burst's TOPS carrier.
    pub fn carrier(&self) -> &TopsCarrier {
        &self.carrier
    }

    /// Bistatic azimuth delay (s) at slant range `range`.
    pub fn bistatic_delay(&self, range: f64) -> f64 {
        self.bistatic_offset + range / SPEED_OF_LIGHT
    }

    /// Azimuth FM-rate mismatch (s) for a target imaged at `t` / `range`.
    pub fn fm_rate_mismatch(
        &self,
        t: f64,
        range: f64,
        target: Vec3,
        platform: &PlatformState,
    ) -> f64 {
        let tau = 2.0 * range / SPEED_OF_LIGHT;
        let ka = self.fm_track.eval(t, tau);
        let ks = 2.0 * norm(platform.velocity) / SPEED_OF_LIGHT
            * self.radar_frequency
            * self.steering_rate_rad;
        let kt = ka * ks / (ka - ks);
        let fdc_geometric = self.dc_track.eval(t, tau);
        let fdc = fdc_geometric
            + kt * ((t - self.burst_mid_time)
                + (self.burst_mid_doppler_over_fm - fdc_geometric / ka));
        let look = sub(platform.position, target);
        let ka_true = -2.0 * self.radar_frequency / (SPEED_OF_LIGHT * norm(look))
            * (dot(look, platform.acceleration) + dot(platform.velocity, platform.velocity));
        fdc * (1.0 / ka_true - 1.0 / ka)
    }

    /// Fractional burst line of azimuth time `t`.
    pub fn line(&self, t: f64) -> f64 {
        (t - self.first_line_time) / self.azimuth_time_interval
    }

    /// Doppler-induced slant-range shift (m) at `t` / `range`.
    pub fn doppler_range_shift(&self, t: f64, range: f64) -> f64 {
        self.carrier.total_doppler(self.line(t), range) / self.range_chirp_rate * SPEED_OF_LIGHT
            / 2.0
    }

    /// Every correction for a target at ellipsoidal `height` seen at
    /// `incidence` (rad from the ellipsoid normal).
    pub fn at(
        &self,
        t: f64,
        range: f64,
        target: Vec3,
        platform: &PlatformState,
        height: f64,
        incidence: f64,
    ) -> Correction {
        let azimuth =
            self.bistatic_delay(range) + self.fm_rate_mismatch(t, range, target, platform);
        let troposphere =
            STATIC_ZENITH_DELAY / incidence.cos() * (-height / STATIC_SCALE_HEIGHT).exp();
        Correction {
            azimuth_time: -azimuth,
            slant_range: self.doppler_range_shift(t, range) + troposphere,
        }
    }
}
