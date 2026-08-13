"""Forward models: shape in, the two measured curves out.

Each model here is independent and can be swapped for any other. They differ in how the body
is parameterised and in which physics they include, so a method can be tried against all of
them.

    convex_egi          Analytic convex operator on the extended Gaussian image. Exact for a
                        convex body, with closed-form adjoints. No shadow, no interreflection.
                        The only one fast enough for a linear solver.

    polytope_raycast    Body as a union of convex polytopes, rendered by exact ray casting.
                        Occlusion and cast shadow, single bounce. No mesh and no rasteriser,
                        so it runs on a GPU without one.

    sdf_surface         Body as a level set, first hit by sphere tracing. Occlusion and cast
                        shadow, single bounce. Derivatives come from the implicit function
                        theorem, so there is no smoothing bias, but the silhouette is a step
                        and gives no gradient through the outline.

    sdf_volumetric      The same level set, mollified and integrated along the ray. Visibility
                        and shadowing become smooth functions of the geometry, which is what
                        the hard-surface model lacks, at the cost of a bias that fills necks
                        and rounds edges. Intended as a continuation stage, annealed in w.

    mesh/               The full physical chain, in three parts: radiosity (interreflection),
                        raster (perspective rasterisation and the reduction to curves) and
                        sensor (cos^4, vignetting, PSF, OETF, clipping, 8-bit quantisation).
                        The only model here with multiple bounces and a sensor.

    learned_surrogate   A fast approximation of the mesh chain: geometry is ray-traced exactly
                        and only the tone-mapped response is learned. Built for the thousands
                        of evaluations an unrolled solver needs. Its agreement with the mesh
                        chain is the thing to check before relying on it.

    shared/             Not models: the common contract and the two reductions (common), exact
                        derivatives through the image thresholds (coarea), and the nvdiffrast
                        loader (_nvdr).

Every model takes directions already carried into the body frame by hac26.conventions.to_body,
and returns the intensity and lit-area pair defined in shared.common.
"""
