'''
In this module, sampled data is analyzed eith respect to bounds and a convex
Polytope is developed using a H-representation of the upper and lower bounds for
each independent variable and the ratios in-between the independent variables.
The V-representation can then be obtained using the package pyCDDLIB.
Subseqeuently, Delaunay triangulation is applied for the identification of the
n-simplices of the polytope. If specified, random point placement in the
polytope can be conducted based on the respective volume fraction to assure
uniform distribution of the sampling points.

The object GridPolytope has the following functions:
    .DataAna:       Analysis of the provided data, bounds calculation
    .Polytope_Cal:  Calculation of the H- and V-representation of the polytope
    .Simplex_Cal:   Calculation of n-simplex properties
    .Tri_Cal:       Point placement in a polygon    (through triangles)
    .Tetra_Cal:     Point placement in a polyhedron (through tetrahedrons)
    .Simp_Cal:      Point placement in a polytope (through n-simplices)

Tri_Cal and Tetra_Cal use folding of the points in the parallelogram and
parallelepiped respectively, while Simp_Cal places randomly points in one dimension
and then uses probabilty distribution to sample new points in the other dimensions.
It can be applied to an arbitrary number of dimensions.
'''

# Import of the required packages
import numpy as np
import cdd
from scipy.spatial import Delaunay

class SimplexClass:
    '''
    Class definition of a n-simplex.
    '''
    def __init__(self):
        self.V     = 0
        self.Vfrac = 0

class GridPolytope:
    def __init__(self):
        self.Bounds = None
        self.Vert_Poly = None

    def Data_Ana(self,Sample,ind_flow,ind_ratio,ind_sum):
        '''
        Function for the calculation of the calculation of several properties of
        the sampled data. This includes the upper and lower bounds of both the
        the flowrates and ratios based on the calculated data

        Additional input to the functions are provided as
            Sample:     Sample space data from the previous unit operations
            ind_flow:   Tuple of the indices of the flowrates in "sample" one
                        wants to investigate
            ind_ratio:  2D tuple in which each row corresponds to a ratio.
            ind_sum:    2D tuple in which each row corresponds to a sum.

        The calculated properties include
            .Bounds:    Upper and lower bounds of the flowrates and ratios
                        with the order of the bounds is given by
                        1. Flowrates according to tuple order
                        2. Ratios according to tuple row order
        '''
        # Calculation of temporary variables
        ratio_true = np.array(ind_ratio).ndim == 2
        sum_true   = np.array(ind_sum).ndim == 2
        tmp_1 = np.shape(ind_flow)[0]
        tmp_2 = np.shape(ind_ratio)[0] if ratio_true else 1
        tmp_3 = np.shape(ind_sum)[0] if sum_true else 1

        # Adjustment for empty lists
        tmp_2 = tmp_2 if ind_ratio else 0
        tmp_3 = tmp_3 if ind_sum else 0
        
        # Calculation of the flowrate ratios
        ratio = np.zeros((Sample.shape[0],tmp_2))
        for k in range(tmp_2):
            tmp = ind_ratio[slice(k,k+1)][0] if ratio_true else ind_ratio
            ratio[:,k] = Sample[:,tmp[0]]/Sample[:,tmp[1]]

        # Calculation of the sums of the different flow rates
        flow_sum = np.zeros((Sample.shape[0],tmp_3))
        for k in range(tmp_3):
            tmp = ind_sum[slice(k,k+1)][0] if sum_true else ind_sum
            flow_sum[:,k] = Sample[:,tmp[0]]+Sample[:,tmp[1]]
            
        self.ratio = ratio
        self.sum   = flow_sum

        # Calculation of the Bounds
        self.Bounds = np.ones((2,tmp_1+tmp_2+tmp_3))
        self.Bounds[0,0:tmp_1]           = Sample[:,ind_flow].min(axis = 0)
        self.Bounds[1,0:tmp_1]           = Sample[:,ind_flow].max(axis = 0)
        self.Bounds[0,tmp_1:tmp_1+tmp_2] = ratio.min(axis = 0)
        self.Bounds[1,tmp_1:tmp_1+tmp_2] = ratio.max(axis = 0)
        self.Bounds[0,tmp_1+tmp_2:]      = flow_sum.min(axis = 0)
        self.Bounds[1,tmp_1+tmp_2:]      = flow_sum.max(axis = 0)
            
    def Polytope_Cal(self,Sample,ind_flow,ind_ratio,ind_sum):
        '''
        Function for calculating the vertices of the polytope. This is
        achieved by first calculating the upper and lower bounds on the flowrates
        and ratios using the function "Data_Ana", if this is not conducted
        beforehand, and subsequently applying the double description method for
        the vertex enumeration problem using the inequalities given by the 
        ratios. This method is implemented in the library pycDDLIB.

        Additional input to the functions are provided as
            ind_flow:   Tuple of the indices of the flowrates one wants to 
                        investigate.
            ind_ratio:  2D tuple in which each row corresponds to a ratio.

        The calculated properties include
            .InEq:      Inequality constraints defining the polytope
            .Vert_Poly: Vertices of the polytope defined by the inequality
                        constraints
        '''
        # Calculation of the bounds if this is not done beforehand
        if self.Bounds is None:
            self.Data_Ana(Sample,ind_flow,ind_ratio,ind_sum)

        # Calculation of temporary variables
        ratio_true = np.array(ind_ratio).ndim == 2
        sum_true   = np.array(ind_sum).ndim == 2
        tmp_1 = np.shape(ind_flow)[0]
        tmp_2 = np.shape(ind_ratio)[0] if ratio_true else 1
        tmp_3 = np.shape(ind_sum)[0] if sum_true else 1
        
        # Adjustment for empty lists
        tmp_2 = tmp_2 if ind_ratio else 0
        tmp_3 = tmp_3 if ind_sum else 0

        # Calculation of the inequality constraints in the form b+A \geq 0
        InEq  = -np.zeros(((tmp_1+tmp_2+tmp_3)*2,tmp_1+1))
        for k in range(tmp_1+tmp_2+tmp_3):
            if k < tmp_1:
                # Lower Bound on each flow variable
                InEq[2*k,0]     = -self.Bounds[0,k]
                InEq[2*k,k+1]   =  1
                # Upper bound on each flow variable
                InEq[2*k+1,0]   =  self.Bounds[1,k]
                InEq[2*k+1,k+1] = -1
            elif k < tmp_1+tmp_2:
                # Extraction of the index
                tmp_3 = ind_ratio[slice(k-tmp_1,k+1-tmp_1)][0][1] \
                        if ratio_true else ind_ratio[1]
                tmp_4 = ind_ratio[slice(k-tmp_1,k+1-tmp_1)][0][0] \
                        if ratio_true else ind_ratio[0]
                # Bound defined by the minimum ratio
                InEq[2*k,ind_flow.index(tmp_3)+1]   = -self.Bounds[0,k]
                InEq[2*k,ind_flow.index(tmp_4)+1]   =  1
                # Bound defined by the maximum ratio
                if not np.isinf(self.Bounds[1,k]):
                    InEq[2*k+1,ind_flow.index(tmp_3)+1] =  self.Bounds[1,k]
                    InEq[2*k+1,ind_flow.index(tmp_4)+1] = -1
            else:
                # Extraction of the index
                tmp_3 = ind_sum[slice(k-tmp_1-tmp_2,k+1-tmp_1-tmp_2)][0][1] \
                        if sum_true else ind_sum[1]
                tmp_4 = ind_sum[slice(k-tmp_1-tmp_2,k+1-tmp_1-tmp_2)][0][0] \
                        if sum_true else ind_sum[0]

                # Bound defined by the minimum sum
                InEq[2*k,0]                       = -self.Bounds[0,k]
                InEq[2*k,ind_flow.index(tmp_3)+1] =  1
                InEq[2*k,ind_flow.index(tmp_4)+1] =  1
                # Bound defined by the maximum sum
                InEq[2*k+1,0]                       =  self.Bounds[1,k]
                InEq[2*k+1,ind_flow.index(tmp_3)+1] = -1
                InEq[2*k+1,ind_flow.index(tmp_4)+1] = -1
        self.InEq = InEq
        self.InEq_A = -InEq[:,1:]
        self.InEq_b = InEq[:,0]
        
        # Calculation of the vertices of the problem using the package pyCDDLib
        mat = cdd.Matrix(self.InEq)
        mat.rep_type = cdd.RepType.INEQUALITY
        poly = cdd.Polyhedron(mat)
        ext  = poly.get_generators()
        Vert_Poly = np.asarray(ext.__getitem__(slice(0,ext.row_size)))
        self.Vert_Poly = Vert_Poly[:,1:tmp_1+1]

        # Calculation of the indices of the simplices formed by the vertices
        # of the polytope, assignment of the values of the triangles or
        # simplices, calculation of the overall area or volume, and the
        # area/volume fractions of the individual triangles/simplices
        Vert_Ind = Delaunay(self.Vert_Poly)
        self.Vert_Ind = Vert_Ind
        self.Simplex = [SimplexClass() for _ in range(Vert_Ind.nsimplex)]
        
        for k in range(Vert_Ind.nsimplex):
            self.Simplex[k].Vert = self.Vert_Poly[(Vert_Ind.simplices[k]),:]
            self.Simplex_Cal(k)

        self.V = np.sum(self.Simplex[k].V for k in range(Vert_Ind.nsimplex))
        for k in range (Vert_Ind.nsimplex): 
            self.Simplex[k].Vfrac = self.Simplex[k].V/self.V

    def Simplex_Cal(self,k):
        '''
        Function for the calculation of the different properties of a simplex.
        These properties include the barycentric representation and volume of
        the simplex.
        
        Additional input to the functions are provided as
            k:          Index of the tetrahedron as calculated using Delaunay
                        tesselation
        
        The calculated properties include (based on self.Simplex[k]):
            .Vert_BC:   Barycentric coordinates using the first vertice as 
                        reference
            .V:         Volume of the simplex (if triangle, it is the area)
        '''
        # Calculation of the barycentric coordinates and the volume
        self.Simplex[k].Vert_BC = self.Simplex[k].Vert[1:,:]-self.Simplex[k].Vert[0,:]
        self.Simplex[k].V = np.absolute(np.linalg.det(self.Simplex[k].Vert_BC.transpose())/np.math.factorial(self.Vert_Poly.shape[1]))
        
    def Simp_Point(self,**kwargs):
        '''
        Function for the calculation of the sampling domain within the polytope
        defined through the n-simplices. It requires that Simplex_Cal, and
        therefore, Polytope_Cal is run beforehand.

        Additional input to the functions are provided with the keywords
            domain:     n_samp x ind_flow random distributed points.
            n_samp:     Number of sampling ponts

        The function returns:
            domain_tri: n_samp x ind_flow random distributed points inside the
                        triangle.
        '''

        # Assignment of the domain
        if "domain" in kwargs:
            domain = kwargs["domain"]
            n_samp = domain.shape[0]
            n_flow = domain.shape[1]
        
        elif "n_samp" in kwargs and "n_flow" in kwargs:
            n_samp = kwargs["n_samp"]
            n_flow = kwargs["n_flow"]
            domain = np.random.rand(n_samp,n_flow)
        else:
            raise ValueError('Neither a sampling domain (domain) was provided,'\
                             'nor a number of sampling points (n_samp) and '\
                             'the number of flows (n_flow).')
        # Calculation of the distribution within the n-simplex
        rand_numb = [0]*n_flow
        Ex_Mat = np.eye(n_flow)
        for k in range(n_flow):
            rand_numb[k] = domain[:,k].reshape(-1,1)**(1/(k+1))
        domain = rand_numb[0]*np.zeros((1,n_flow))+\
                 (1-rand_numb[0])*Ex_Mat[-1,:].reshape(1,-1)
        for k in range(1,n_flow):
            domain = rand_numb[k]*domain+\
                     (1-rand_numb[k])*Ex_Mat[-1-k,:].reshape(1,-1)
        
        # Assignment of the points to the different simplices
        dec_simplex = np.random.rand(n_samp)
        domain_tri = np.zeros((n_samp,n_flow))
        for k in range(n_samp):
            tmp_1 = 0
            tmp_2 = self.Simplex[0].Vfrac
            for l in range(self.Vert_Ind.nsimplex):
                if tmp_1 <= dec_simplex[k] < tmp_2:
                    break
                else:
                    tmp_1 = tmp_2
                    tmp_2 = tmp_2+self.Simplex[l+1].Vfrac
            domain_tri[k,:] = self.Simplex[l].Vert[0,:] + \
                              np.sum(domain[k,m]*self.Simplex[l].Vert_BC[m,:] \
                              for m in range(n_flow))
        return domain_tri
