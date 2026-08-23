//! Linear source-influence graph and exact covariance contraction.
//!
//! Nodes store local Jacobians only. Covariance queries traverse those
//! Jacobians in reverse and combine all paths that reach the same primitive
//! source before contracting that source.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

use dolphin_core::Cf64;
use ndarray::{Array2, ArrayView2};

/// Stable identifier for one primitive stochastic source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceId(u64);

impl SourceId {
    /// Construct a source identifier from its persistent integer value.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the persistent integer value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Stable identifier for one derived influence node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u64);

impl NodeId {
    /// Construct a node identifier from its persistent integer value.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the persistent integer value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Definition of one independent primitive source vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceDefinition {
    id: SourceId,
    dimension: usize,
    model_hash: [u8; 32],
}

impl SourceDefinition {
    /// Define a source with a fixed tangent dimension and model identity.
    #[must_use]
    pub const fn new(id: SourceId, dimension: usize, model_hash: [u8; 32]) -> Self {
        Self {
            id,
            dimension,
            model_hash,
        }
    }

    /// Source identifier.
    #[must_use]
    pub const fn id(&self) -> SourceId {
        self.id
    }

    /// Number of independent standard-normal source coordinates.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Digest identifying the ordered source-factor model.
    #[must_use]
    pub const fn model_hash(&self) -> &[u8; 32] {
        &self.model_hash
    }
}

/// Linear edge `y_child += coefficient * y_parent`.
#[derive(Debug, Clone)]
pub struct ParentEdge {
    parent: NodeId,
    coefficient: Array2<f64>,
}

impl ParentEdge {
    /// Construct a derived-parent edge.
    #[must_use]
    pub fn new(parent: NodeId, coefficient: Array2<f64>) -> Self {
        Self {
            parent,
            coefficient,
        }
    }
}

/// Linear edge `y_child += coefficient * xi_source`.
#[derive(Debug, Clone)]
pub struct SourceEdge {
    source: SourceId,
    coefficient: Array2<f64>,
}

impl SourceEdge {
    /// Construct a primitive-source edge.
    #[must_use]
    pub fn new(source: SourceId, coefficient: Array2<f64>) -> Self {
        Self {
            source,
            coefficient,
        }
    }
}

/// One derived node and its local influence operators.
#[derive(Debug, Clone)]
pub struct InfluenceNode {
    id: NodeId,
    dimension: usize,
    parents: Vec<ParentEdge>,
    sources: Vec<SourceEdge>,
}

impl InfluenceNode {
    /// Construct a node before attaching its local parent/source operators.
    #[must_use]
    pub fn new(id: NodeId, dimension: usize) -> Self {
        Self {
            id,
            dimension,
            parents: Vec::new(),
            sources: Vec::new(),
        }
    }

    /// Add one derived-parent operator.
    #[must_use]
    pub fn with_parent(mut self, edge: ParentEdge) -> Self {
        self.parents.push(edge);
        self
    }

    /// Add one primitive-source operator.
    #[must_use]
    pub fn with_source(mut self, edge: SourceEdge) -> Self {
        self.sources.push(edge);
        self
    }

    /// Node identifier.
    #[must_use]
    pub const fn id(&self) -> NodeId {
        self.id
    }

    /// Node vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
}

/// One retained temporal coordinate or the literal acquisition-0 gauge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalCoordinate {
    /// Deterministic acquisition-0 coordinate, represented by an exact zero.
    Gauge,
    /// One component of a derived node.
    Node {
        /// Derived node identifier.
        node: NodeId,
        /// Zero-based component within the node vector.
        component: usize,
    },
}

impl TemporalCoordinate {
    /// Select one component from a derived node.
    #[must_use]
    pub const fn node(node: NodeId, component: usize) -> Self {
        Self::Node { node, component }
    }
}

/// Validation or query failure for a source-influence graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InfluenceError {
    /// A source or node declared a zero-dimensional vector.
    ZeroDimension,
    /// A primitive source used the all-zero missing model digest.
    MissingModelHash,
    /// A source identifier was registered more than once.
    DuplicateSource(SourceId),
    /// A node identifier was registered more than once.
    DuplicateNode(NodeId),
    /// A node edge referred to an unregistered primitive source.
    UnknownSource(SourceId),
    /// A parent edge or query referred to an unregistered node.
    UnknownNode(NodeId),
    /// A query selected a component outside the node vector.
    ComponentOutOfBounds {
        /// Selected node.
        node: NodeId,
        /// Requested component.
        component: usize,
        /// Node vector dimension.
        dimension: usize,
    },
    /// A local operator did not match `(child dimension, parent dimension)`.
    ShapeMismatch,
    /// A local operator contained NaN or infinity.
    NonFiniteOperator,
    /// Reverse propagation or covariance contraction overflowed or became non-finite.
    NonFiniteContraction,
}

impl Display for InfluenceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "influence vectors must have positive dimension"),
            Self::MissingModelHash => write!(f, "primitive source has no model hash"),
            Self::DuplicateSource(id) => write!(f, "duplicate source {}", id.get()),
            Self::DuplicateNode(id) => write!(f, "duplicate node {}", id.get()),
            Self::UnknownSource(id) => write!(f, "unknown source {}", id.get()),
            Self::UnknownNode(id) => write!(f, "unknown or nonpreceding node {}", id.get()),
            Self::ComponentOutOfBounds {
                node,
                component,
                dimension,
            } => write!(
                f,
                "node {} component {component} is outside dimension {dimension}",
                node.get()
            ),
            Self::ShapeMismatch => write!(f, "influence operator shape mismatch"),
            Self::NonFiniteOperator => write!(f, "influence operator is non-finite"),
            Self::NonFiniteContraction => {
                write!(f, "influence covariance contraction is non-finite")
            }
        }
    }
}

impl Error for InfluenceError {}

/// Validation failure for a caller-supplied proper-complex source factor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceModelError {
    /// The factor or ordered component list was empty.
    EmptyFactor,
    /// The component count did not match the square complex factor.
    ComponentCountMismatch,
    /// The ordered component list repeated an identity.
    DuplicateComponent(u64),
    /// The source-model digest was the all-zero missing value.
    MissingModelHash,
    /// The complex factor contained NaN or infinity.
    NonFiniteFactor,
    /// A coefficient above the diagonal was nonzero.
    NotLowerTriangular,
    /// A diagonal coefficient was not positive and real.
    NonPositiveRealDiagonal,
    /// The canonical real source dimension overflowed.
    DimensionOverflow,
    /// A raw-coordinate Jacobian did not have `2n` finite input columns.
    JacobianShapeMismatch,
    /// A raw-coordinate Jacobian contained NaN or infinity.
    NonFiniteJacobian,
    /// Binding a Jacobian through the factor overflowed or became non-finite.
    NonFiniteBoundOperator,
}

impl Display for SourceModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyFactor => write!(f, "proper-complex factor is empty"),
            Self::ComponentCountMismatch => {
                write!(f, "component count does not match proper-complex factor")
            }
            Self::DuplicateComponent(id) => write!(f, "duplicate source component {id}"),
            Self::MissingModelHash => write!(f, "proper-complex factor has no model hash"),
            Self::NonFiniteFactor => write!(f, "proper-complex factor is non-finite"),
            Self::NotLowerTriangular => write!(f, "proper-complex factor is not lower triangular"),
            Self::NonPositiveRealDiagonal => {
                write!(f, "proper-complex factor diagonal is not positive and real")
            }
            Self::DimensionOverflow => write!(f, "proper-complex real dimension overflowed"),
            Self::JacobianShapeMismatch => {
                write!(
                    f,
                    "raw-coordinate Jacobian does not match proper-complex factor"
                )
            }
            Self::NonFiniteJacobian => write!(f, "raw-coordinate Jacobian is non-finite"),
            Self::NonFiniteBoundOperator => {
                write!(f, "factor-bound source operator is non-finite")
            }
        }
    }
}

impl Error for SourceModelError {}

/// Validated proper-complex lower factor with a strong source-model identity.
#[derive(Debug, Clone)]
pub struct ProperComplexFactor {
    source: SourceId,
    component_ids: Vec<u64>,
    model_hash: [u8; 32],
    lower: Array2<Cf64>,
}

impl ProperComplexFactor {
    /// Validate and construct a square proper-complex lower factor.
    ///
    /// # Errors
    /// Returns an error for missing/duplicate component identity, an absent
    /// model digest, a non-finite coefficient, a nonzero upper triangle, or a
    /// non-positive-real diagonal.
    pub fn new(
        source: SourceId,
        component_ids: Vec<u64>,
        model_hash: [u8; 32],
        lower: Array2<Cf64>,
    ) -> Result<Self, SourceModelError> {
        let n = component_ids.len();
        if n == 0 {
            return Err(SourceModelError::EmptyFactor);
        }
        if lower.dim() != (n, n) {
            return Err(SourceModelError::ComponentCountMismatch);
        }
        let mut ordered = component_ids.clone();
        ordered.sort_unstable();
        if let Some(duplicate) = ordered.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(SourceModelError::DuplicateComponent(duplicate[0]));
        }
        if model_hash.iter().all(|byte| *byte == 0) {
            return Err(SourceModelError::MissingModelHash);
        }
        if lower.iter().any(|value| !value.is_finite()) {
            return Err(SourceModelError::NonFiniteFactor);
        }
        for row in 0..n {
            if lower[(row, row)].im != 0.0 || lower[(row, row)].re <= 0.0 {
                return Err(SourceModelError::NonPositiveRealDiagonal);
            }
            if ((row + 1)..n).any(|column| lower[(row, column)] != Cf64::new(0.0, 0.0)) {
                return Err(SourceModelError::NotLowerTriangular);
            }
        }
        Ok(Self {
            source,
            component_ids,
            model_hash,
            lower,
        })
    }

    /// Primitive source identifier.
    #[must_use]
    pub const fn source(&self) -> SourceId {
        self.source
    }

    /// Ordered raw-complex component identities.
    #[must_use]
    pub fn component_ids(&self) -> &[u64] {
        &self.component_ids
    }

    /// Source-factor model digest.
    #[must_use]
    pub const fn model_hash(&self) -> &[u8; 32] {
        &self.model_hash
    }

    /// Validated complex lower factor.
    #[must_use]
    pub const fn lower(&self) -> &Array2<Cf64> {
        &self.lower
    }

    /// Canonical real embedding of the proper-complex factor.
    ///
    /// For complex dimension `n`, this returns the `2n x 2n` matrix
    /// `1/sqrt(2) [[Re(L), -Im(L)], [Im(L), Re(L)]]`.
    #[must_use]
    pub fn real_embedding(&self) -> Array2<f64> {
        let n = self.lower.nrows();
        let scale = std::f64::consts::FRAC_1_SQRT_2;
        Array2::from_shape_fn((2 * n, 2 * n), |(row, column)| {
            let value = match (row < n, column < n) {
                (true, true) => self.lower[(row, column)].re,
                (true, false) => -self.lower[(row, column - n)].im,
                (false, true) => self.lower[(row - n, column)].im,
                (false, false) => self.lower[(row - n, column - n)].re,
            };
            value * scale
        })
    }

    /// Build the only source definition compatible with this factor identity.
    ///
    /// The source dimension is the canonical `2n` real embedding dimension and
    /// the definition reuses this validated factor's source ID and model hash.
    ///
    /// # Errors
    /// Returns an error if the real dimension overflows `usize`.
    pub fn source_definition(&self) -> Result<SourceDefinition, SourceModelError> {
        let dimension = self
            .lower
            .nrows()
            .checked_mul(2)
            .ok_or(SourceModelError::DimensionOverflow)?;
        Ok(SourceDefinition::new(
            self.source,
            dimension,
            self.model_hash,
        ))
    }

    /// Bind an output Jacobian in raw `[Re, Im]` coordinates to this factor.
    ///
    /// `raw_jacobian` has shape `(output_dimension, 2n)`. The returned source
    /// edge is `raw_jacobian * real_embedding()` and therefore carries this
    /// factor's source identity and stochastic coordinates together.
    ///
    /// # Errors
    /// Returns an error for an empty/mismatched/non-finite Jacobian or a
    /// non-finite bound operator.
    pub fn bind_real_jacobian(
        &self,
        raw_jacobian: ArrayView2<f64>,
    ) -> Result<SourceEdge, SourceModelError> {
        let input_dimension = self
            .lower
            .nrows()
            .checked_mul(2)
            .ok_or(SourceModelError::DimensionOverflow)?;
        if raw_jacobian.nrows() == 0 || raw_jacobian.ncols() != input_dimension {
            return Err(SourceModelError::JacobianShapeMismatch);
        }
        if raw_jacobian.iter().any(|value| !value.is_finite()) {
            return Err(SourceModelError::NonFiniteJacobian);
        }
        let coefficient = raw_jacobian.dot(&self.real_embedding());
        if coefficient.iter().any(|value| !value.is_finite()) {
            return Err(SourceModelError::NonFiniteBoundOperator);
        }
        Ok(SourceEdge::new(self.source, coefficient))
    }
}

/// A validated acyclic graph of local linear source influences.
#[derive(Debug, Default, Clone)]
pub struct InfluenceDag {
    sources: BTreeMap<SourceId, SourceDefinition>,
    nodes: BTreeMap<NodeId, InfluenceNode>,
    node_order: Vec<NodeId>,
}

impl InfluenceDag {
    /// Construct an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one independent primitive source.
    ///
    /// # Errors
    /// Returns an error for a zero dimension, missing model hash, or duplicate identifier.
    pub fn add_source(&mut self, source: SourceDefinition) -> Result<(), InfluenceError> {
        if source.dimension == 0 {
            return Err(InfluenceError::ZeroDimension);
        }
        if source.model_hash.iter().all(|byte| *byte == 0) {
            return Err(InfluenceError::MissingModelHash);
        }
        if self.sources.contains_key(&source.id) {
            return Err(InfluenceError::DuplicateSource(source.id));
        }
        self.sources.insert(source.id, source);
        Ok(())
    }

    /// Append one node after all of its parents and sources are registered.
    ///
    /// # Errors
    /// Returns an error for duplicate identifiers, unresolved dependencies,
    /// invalid dimensions, shape mismatches, or non-finite coefficients.
    pub fn add_node(&mut self, node: InfluenceNode) -> Result<(), InfluenceError> {
        self.validate_node(&node)?;
        self.node_order.push(node.id);
        self.nodes.insert(node.id, node);
        Ok(())
    }

    /// Reconstruct selected temporal covariance by reverse source contraction.
    ///
    /// Gauge coordinates are exact zeros. All reverse paths that reach the
    /// same source are summed before that source contributes `Z'Z`.
    ///
    /// # Errors
    /// Returns an error when a selected node or component does not exist.
    pub fn temporal_covariance(
        &self,
        coordinates: &[TemporalCoordinate],
    ) -> Result<Array2<f64>, InfluenceError> {
        let selected = coordinates.len();
        let mut node_adjoints: BTreeMap<NodeId, Array2<f64>> = BTreeMap::new();
        for (column, coordinate) in coordinates.iter().copied().enumerate() {
            let TemporalCoordinate::Node { node, component } = coordinate else {
                continue;
            };
            let definition = self
                .nodes
                .get(&node)
                .ok_or(InfluenceError::UnknownNode(node))?;
            if component >= definition.dimension {
                return Err(InfluenceError::ComponentOutOfBounds {
                    node,
                    component,
                    dimension: definition.dimension,
                });
            }
            node_adjoints
                .entry(node)
                .or_insert_with(|| Array2::zeros((definition.dimension, selected)))
                [(component, column)] += 1.0;
        }

        let mut source_adjoints: BTreeMap<SourceId, Array2<f64>> = BTreeMap::new();
        for node_id in self.node_order.iter().rev() {
            let Some(adjoint) = node_adjoints.remove(node_id) else {
                continue;
            };
            let node = &self.nodes[node_id];
            for edge in &node.sources {
                let propagated = edge.coefficient.t().dot(&adjoint);
                ensure_finite_contraction(&propagated)?;
                let root = source_adjoints.entry(edge.source).or_insert_with(|| {
                    Array2::zeros((self.sources[&edge.source].dimension, selected))
                });
                *root += &propagated;
                ensure_finite_contraction(root)?;
            }
            for edge in &node.parents {
                let propagated = edge.coefficient.t().dot(&adjoint);
                ensure_finite_contraction(&propagated)?;
                let parent_dimension = self.nodes[&edge.parent].dimension;
                let parent = node_adjoints
                    .entry(edge.parent)
                    .or_insert_with(|| Array2::zeros((parent_dimension, selected)));
                *parent += &propagated;
                ensure_finite_contraction(parent)?;
            }
        }

        let mut covariance = Array2::zeros((selected, selected));
        for root in source_adjoints.values() {
            let contribution = root.t().dot(root);
            ensure_finite_contraction(&contribution)?;
            covariance += &contribution;
            ensure_finite_contraction(&covariance)?;
        }
        Ok(covariance)
    }

    /// Number of registered primitive sources.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Number of registered derived nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn validate_node(&self, node: &InfluenceNode) -> Result<(), InfluenceError> {
        if node.dimension == 0 {
            return Err(InfluenceError::ZeroDimension);
        }
        if self.nodes.contains_key(&node.id) {
            return Err(InfluenceError::DuplicateNode(node.id));
        }
        for edge in &node.parents {
            let parent = self
                .nodes
                .get(&edge.parent)
                .ok_or(InfluenceError::UnknownNode(edge.parent))?;
            validate_operator(&edge.coefficient, node.dimension, parent.dimension)?;
        }
        for edge in &node.sources {
            let source = self
                .sources
                .get(&edge.source)
                .ok_or(InfluenceError::UnknownSource(edge.source))?;
            validate_operator(&edge.coefficient, node.dimension, source.dimension)?;
        }
        Ok(())
    }
}

fn ensure_finite_contraction(matrix: &Array2<f64>) -> Result<(), InfluenceError> {
    match matrix.iter().all(|value| value.is_finite()) {
        true => Ok(()),
        false => Err(InfluenceError::NonFiniteContraction),
    }
}

fn validate_operator(
    operator: &Array2<f64>,
    output_dimension: usize,
    input_dimension: usize,
) -> Result<(), InfluenceError> {
    if operator.dim() != (output_dimension, input_dimension) {
        return Err(InfluenceError::ShapeMismatch);
    }
    if operator.iter().any(|value| !value.is_finite()) {
        return Err(InfluenceError::NonFiniteOperator);
    }
    Ok(())
}
