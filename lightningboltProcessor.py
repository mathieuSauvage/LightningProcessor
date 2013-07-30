#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

'''
================================================================================
* VERSION 0.0
================================================================================
* AUTHOR:
Mathieu Sauvage mathieu@hiddenforest.fr
================================================================================
* INTERNET SOURCE:
================================================================================
* DESCRIPTION:
This is a lightning Processor designed to process batch of lightnings and let a
good control over the shape of the lightnings.  It uses numpy to try to have
a python code as efficient as possible
================================================================================
* USAGE:
================================================================================
* TODO:
- implement the time accumulator
- finish the Ring lightning
- generate full rotation matrix instead of just vector on sphere ( so we can transmit the frame and we don't need to calculate the start Up for every branch or maybe not)
================================================================================
'''

import numpy as np
import math
import random
import sys
import time

class LBP_msCommandException(Exception):
	def __init__(self,message):
		self.message = '[LBP] '+message
	
	def __str__(self):
		return self.message

def enum(*sequential, **named):
	enums = dict(zip(sequential, range(len(sequential))), **named)
	return type('Enum', (), enums)

class avgTimer:
	def __init__( self, numValues ):
		self.arrayOfTime = [0.0]*numValues
		self.nextIndex = 0
		self.startTime = 0.0

	def average(self):
		return sum(self.arrayOfTime)/float(len(self.arrayOfTime))

	def start( self ):
		self.startTime = time.clock()

	def stop( self ):
		self.arrayOfTime[self.nextIndex] = time.clock() - self.startTime
		self.nextIndex = (self.nextIndex + 1 ) % len(self.arrayOfTime)
		return self.average()

def generateRandomParams( seeds ):
	pass

def dot2d(g, x, y):
	return g[0]*x + g[1]*y

_CONST1 = 0.5 * (math.sqrt(3.0) - 1.0)
_CONST2 = (3.0 - math.sqrt(3.0)) / 6.0

class simpNoise():
	'''
	noise algorithm transformed from
	http://www.6by9.net/simplex-noise-for-c-and-python/
	'''
	"""The gradients are the midpoints of the vertices of a cube."""

	permutation = np.array([151,160,137,91,90,15, 
		131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23, 
		190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33, 
		88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166, 
		77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244, 
		102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196, 
		135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123, 
		5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42, 
		223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9, 
		129,22,39,253,9,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228, 
		251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107, 
		49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254, 
		138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180], np.int )

	_grad3 = np.array([
	[1,1,0], [-1,1,0], [1,-1,0], [-1,-1,0],
	[1,0,1], [-1,0,1], [1,0,-1], [-1,0,-1],
	[0,1,1], [0,-1,1], [0,1,-1], [0,-1,-1]], np.int)

	@staticmethod
	def generatePermutationTables( seed, tableDimension ):
		'''
		Permutation tables are like the seed of the Simplex noise
		tableDimension, is the maximum dimension of the output you can get with the returned permutation tables (to output a vector tableDimension should be at least 3)
		'''
		nprndState = np.random.RandomState( int(seed) )
		perm = np.tile( simpNoise.permutation , (tableDimension,1) )
		for i in range(tableDimension):
			nprndState.shuffle( perm[i] )
		return np.tile( perm , (2,1,1) ).transpose(1,0,2).reshape(tableDimension,-1)

	@staticmethod
	def in2D_outVectDim( noiseCoords, permMat ):
		numValues = len(noiseCoords)
		outputVectorDimension = len(permMat)
		
		# Skew the input space to determine which simplex cell we're in
		# Hairy skew factor for 2D
		ij2 = noiseCoords.copy()
		s2 = noiseCoords.sum(-1)*_CONST1#------------
		ij2.transpose((1,0))[:] += s2#------------
		ij2 = ij2.astype(int)#------------

		t2 = ij2.sum(-1).astype(np.float32) * _CONST2 #------------

		# Work out the hashed gradient indices of the three simplex corners
		ii = ij2[:,0] & 255# i & 255
		jj = ij2[:,1] & 255#j & 255

		# Unskew the cell origin back to (x,y) space
		XY02 = ij2.astype(np.float32)#------------
		XY02.transpose((1,0))[:] -= t2#------------

		# The x,y distances from the cell origin
		#coord02 = noiseCoords.copy()
		#coord02 -= XY02#------------

		coord = np.tile( noiseCoords - XY02, (3,1,1) )
		#sys.stderr.write('coord '+str( coord  ) +'\n' )

		# For the 2D case, the simplex shape is an equilateral triangle.
		# Determine which simplex we are in.
		
		# Offsets for second (middle) corner of simplex in (i,j) coords
		positiveIds2 = np.where( coord[0,:,0]>coord[0,:,1] )#------------
		ij12 = np.tile( [0,1], (numValues,1))#------------
		ij12[positiveIds2] = [1,0]#------------

		coord[1] -= ij12 - _CONST2
		coord[2] += (2.0 * _CONST2 - 1.0)

		#sys.stderr.write('coord '+str( coord  ) +'\n' )

		Gcoord = np.tile( coord, (1,1,outputVectorDimension) ).reshape(3,-1,outputVectorDimension,2)
		#sys.stderr.write('Gcoord '+str(Gcoord) +'\n' )

	 	coordMatrix = np.tile( ii, (3,outputVectorDimension,1) )
		#sys.stderr.write('coordMatrix '+str(coordMatrix) +'\n' )
	 	coordMatrix[1,:] += ij12[:,0]
	 	coordMatrix[2,:] += 1		
		coordMatrix[0,:] += permMat[:outputVectorDimension,jj]
		coordMatrix[1,:] += permMat[:outputVectorDimension,jj+ij12[:,1]]
		coordMatrix[2,:] += permMat[:outputVectorDimension,jj+1]
		for i in range(outputVectorDimension):
			coordMatrix[:,i] = [ permMat[i, coordMatrix[0,i] ], permMat[i, coordMatrix[1,i] ], permMat[i, coordMatrix[2,i] ] ]

		coordMatrix = coordMatrix.transpose(0,2,1).reshape(-1,outputVectorDimension)
		coordMatrix %= 12
		#sys.stderr.write('coordMatrix '+str(coordMatrix) +'\n' )

		tGlob = .5 - (coord**2).sum(-1)
		#sys.stderr.write('tGlob '+str(tGlob) +'\n' )
		allIndPos = np.where( tGlob.reshape(-1) >=0.0 ) 
		#sys.stderr.write('allIndPos '+str(allIndPos) +'\n' )
		tGlob.reshape(-1)[allIndPos]**=4 # in place power 4
		#sys.stderr.write('tGlob '+str(tGlob) +'\n' )

		tGrad = simpNoise._grad3[ coordMatrix[allIndPos[0] ] ][:,:,:-1]
		#tGrad = simpNoise._grad3[ gi.reshape(-1,outputVectorDimension)[allIndPos[0]] ][:,:,:-1]
		#sys.stderr.write('tGrad '+str(tGrad) +'\n' )
		
		tPos = tGlob.reshape(-1)[allIndPos[0]]
		#GtPos = np.array( [tPos, tPos, tPos, tPos] ).T
		GtPos = np.tile(tPos, (outputVectorDimension,1) ).T

		# Noise contributions from the three corners
		contribG = np.zeros( (3*numValues,outputVectorDimension), np.float32 )

		contribG[allIndPos[0]] = GtPos* (tGrad*Gcoord.reshape(-1,outputVectorDimension,2)[allIndPos[0]]).sum(-1)

		return contribG.reshape(3,numValues,-1).transpose(1,2,0).sum(-1)*70.0
		#contribG = contribG.reshape(3,numValues,-1).transpose(1,2,0)
		#return 70.0*(contribG.sum(-1))

	@staticmethod
	def in2D_outVect( noiseCoords, permMat ):
		# Noise contributions from the three corners
		contrib = np.zeros( (len(noiseCoords),3,3) , np.float32) # samples, coord(X,Y,Z), contrib
		
		# Skew the input space to determine which simplex cell we're in
		F2 = _CONST1#0.5 * (math.sqrt(3.0) - 1.0)
		# Hairy skew factor for 2D

		s2 = noiseCoords.sum(-1)*F2#------------
		ij2 = noiseCoords.copy()#------------
		ij2.transpose((1,0))[:] += s2#------------
		ij2 = ij2.astype(int)#------------

		G2 = _CONST2#(3.0 - math.sqrt(3.0)) / 6.0
		t2 = ij2.sum(-1).astype(np.float32) * G2 #------------

		# Unskew the cell origin back to (x,y) space
		XY02 = ij2.copy().astype(np.float32)#------------
		XY02.transpose((1,0))[:] -= t2#------------

		# The x,y distances from the cell origin
		coord02 = noiseCoords - XY02#------------

		# For the 2D case, the simplex shape is an equilateral triangle.
		# Determine which simplex we are in.
		
		# Offsets for second (middle) corner of simplex in (i,j) coords
		positiveIds2 = np.where( coord02[:,0]>coord02[:,1] )#------------
		ij12 = np.tile( [0,1], (len(noiseCoords),1))#------------
		ij12[positiveIds2] = [1,0]#------------

		# A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
		# a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
		# c = (3-sqrt(3))/6
		coord12 = coord02 - ij12 + G2#------------
		#x1 = x0 - i1 + G2       # Offsets for middle corner in (x,y) unskewed coords
		#y1 = y0 - j1 + G2

		coord22 = coord02.copy() + (2.0 * G2 - 1.0)#------------
		#x2 = x0 + (2.0 * G2 - 1.0) # Offsets for last corner in (x,y) unskewed coords
		#y2 = y0 + (2.0 * G2 - 1.0)

		Gcoord02 = np.array( [coord02,coord02,coord02] )
		Gcoord02 = Gcoord02.transpose( (1,0,2) )
		#sys.stderr.write('Gcoord02 '+str(Gcoord02) +'\n' )

		Gcoord12 = np.array( [coord12,coord12,coord12] )
		Gcoord12 = Gcoord12.transpose( (1,0,2) )
		#sys.stderr.write('Gcoord12 '+str(Gcoord12) +'\n' )

		Gcoord22 = np.array( [coord22,coord22,coord22] )
		Gcoord22 = Gcoord22.transpose( (1,0,2) )
		#sys.stderr.write('Gcoord22 '+str(Gcoord22) +'\n' )

		# Work out the hashed gradient indices of the three simplex corners
		ii = ij2[:,0] & 255# i & 255
		jj = ij2[:,1] & 255#j & 255

		gi0 = np.array( [ permMat[0,ii+permMat[0,jj]], permMat[1,ii+permMat[1,jj]], permMat[2,ii+permMat[2,jj]] ] ) % 12
		#sys.stderr.write('gi0'+str(gi0) +'\n' )
		gi0 = gi0.T
		#sys.stderr.write('gi0.T'+str(gi0) +'\n' )

		gi1 = np.array( [ permMat[0,ii+ij12[:,0]+permMat[0,jj+ij12[:,1]]], permMat[1,ii+ij12[:,0]+permMat[1,jj+ij12[:,1]]], permMat[2,ii+ij12[:,0]+permMat[2,jj+ij12[:,1]]] ] ) % 12
		#sys.stderr.write('gi1'+str(gi1) +'\n' )
		gi1 = gi1.T
		#sys.stderr.write('gi1T'+str(gi1) +'\n' )

		gi2 = np.array( [ permMat[0,ii+1+permMat[0,jj+1]], permMat[1,ii+1+permMat[1,jj+1]], permMat[2,ii+1+permMat[2,jj+1]]] ) % 12
		gi2 = gi2.T

		# Calculate the contribution from the three corners
		#t0 = 0.5 - x0**2 - y0**2
		t0 = .5 - (coord02**2).sum(-1)
		indPos = np.where( t0>=0.0 )
		t0Pos = t0[indPos]**4
		Gt0Pos = np.array( [t0Pos, t0Pos, t0Pos] )
		Gt0Pos = Gt0Pos.T		
		t = (simpNoise._grad3[ gi0[indPos] ])[:,:,:-1]
		contrib[indPos,:,0] = Gt0Pos* (t*Gcoord02[indPos]).sum(-1)

		t1 = .5 - (coord12**2).sum(-1)
		indPos = np.where( t1>=0.0 )
		t1Pos = t1[indPos]**4
		Gt1Pos = np.array( [t1Pos, t1Pos, t1Pos] )
		Gt1Pos = Gt1Pos.T
		t = (simpNoise._grad3[ gi1[indPos] ])[:,:,:-1]
		contrib[indPos,:,1] = Gt1Pos* (t*Gcoord12[indPos]).sum(-1)
		#sys.stderr.write('contrib2 '+str( contrib[indPos,:,1]  ) +'\n' )

		t2 = .5 - (coord22**2).sum(-1)
		indPos = np.where( t2>=0.0 )
		t2Pos = t2[indPos]**4
		Gt2Pos = np.array( [t2Pos, t2Pos, t2Pos] )
		Gt2Pos = Gt2Pos.T
		t = (simpNoise._grad3[ gi2[indPos] ])[:,:,:-1]
		contrib[indPos,:,2] = Gt2Pos* (t*Gcoord22[indPos]).sum(-1)
		#sys.stderr.write('contrib3 '+str( contrib[indPos,:,2]  ) +'\n' )
		sys.stderr.write('Reference contrib '+str( contrib  ) +'\n' )

		return 70.0*(contrib.sum(-1))

	@staticmethod
	def raw_noise_2d( x, y, permMat):
	    """2D Raw Simplex noise."""
	    #sys.stderr.write('raw '+str(x)+' '+str(y)+'\n')
	    # Noise contributions from the three corners
	    n0, n1, n2 = 0.0, 0.0, 0.0

	    # Skew the input space to determine which simplex cell we're in
	    F2 = 0.5 * (math.sqrt(3.0) - 1.0)
	    # Hairy skew factor for 2D
	    s = (x + y) * F2
	    i = int(x + s)
	    j = int(y + s)

	    G2 = (3.0 - math.sqrt(3.0)) / 6.0
	    t = float(i + j) * G2
	    # Unskew the cell origin back to (x,y) space
	    X0 = i - t
	    Y0 = j - t
	    # The x,y distances from the cell origin
	    x0 = x - X0
	    y0 = y - Y0

	    # For the 2D case, the simplex shape is an equilateral triangle.
	    # Determine which simplex we are in.
	    i1, j1 = 0, 0 # Offsets for second (middle) corner of simplex in (i,j) coords
	    if x0 > y0: # lower triangle, XY order: (0,0)->(1,0)->(1,1)
	        i1 = 1
	        j1 = 0
	    else:        # upper triangle, YX order: (0,0)->(0,1)->(1,1)
	        i1 = 0
	        j1 = 1

	    # A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
	    # a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
	    # c = (3-sqrt(3))/6
	    x1 = x0 - i1 + G2       # Offsets for middle corner in (x,y) unskewed coords
	    y1 = y0 - j1 + G2
	    x2 = x0 - 1.0 + 2.0 * G2  # Offsets for last corner in (x,y) unskewed coords
	    y2 = y0 - 1.0 + 2.0 * G2

	    # Work out the hashed gradient indices of the three simplex corners
	    ii = int(i) & 255
	    jj = int(j) & 255
	    gi0 = permMat[0,ii+permMat[3,jj]] % 12
	    gi1 = permMat[0,ii+i1+permMat[3,jj+j1]] % 12
	    gi2 = permMat[0,ii+1+permMat[3,jj+1]] % 12


	    # Calculate the contribution from the three corners
	    t0 = 0.5 - x0*x0 - y0*y0
	    if t0 < 0:
	        n0 = 0.0
	    else:
	        t0 *= t0
	        n0 = t0 * t0 * dot2d(simpNoise._grad3[gi0], x0, y0)

	    t1 = 0.5 - x1*x1 - y1*y1
	    if t1 < 0:
	        n1 = 0.0
	    else:
	        t1 *= t1
	        n1 = t1 * t1 * dot2d(simpNoise._grad3[gi1], x1, y1)

	    t2 = 0.5 - x2*x2-y2*y2
	    if t2 < 0:
	        n2 = 0.0
	    else:
	        t2 *= t2
	        n2 = t2 * t2 * dot2d(simpNoise._grad3[gi2], x2, y2)

		#sys.stderr.write('n0 n1 n2 '+str(n0)+' '+str(n1)+' '+str(n2)+'\n')

	    # Add contributions from each corner to get the final noise value.
	    # The result is scaled to return values in the interval [-1,1].
	    return 70.0 * (n0 + n1 + n2)

	@staticmethod
	def raw_noise_2dVector( x, y, permMat):
		"""2D Raw Simplex noise."""
		# Noise contributions from the three corners
		n0 = [0.0]*3
		n1 = [0.0]*3
		n2 = [0.0]*3

		#n0, n1, n2 = 0.0, 0.0, 0.0

		# Skew the input space to determine which simplex cell we're in
		F2 = _CONST1#0.5 * (math.sqrt(3.0) - 1.0)
		# Hairy skew factor for 2D
		s = (x + y) * F2
		i = int(x + s)
		j = int(y + s)

		G2 = _CONST2#(3.0 - math.sqrt(3.0)) / 6.0
		t = float(i + j) * G2
		# Unskew the cell origin back to (x,y) space
		X0 = i - t
		Y0 = j - t
		# The x,y distances from the cell origin
		x0 = x - X0
		y0 = y - Y0

		# For the 2D case, the simplex shape is an equilateral triangle.
		# Determine which simplex we are in.
		i1, j1 = 0, 0 # Offsets for second (middle) corner of simplex in (i,j) coords
		if x0 > y0: # lower triangle, XY order: (0,0)->(1,0)->(1,1)
			i1 = 1
			j1 = 0
		else:        # upper triangle, YX order: (0,0)->(0,1)->(1,1)
			i1 = 0
			j1 = 1

		# A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
		# a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
		# c = (3-sqrt(3))/6
		x1 = x0 - i1 + G2       # Offsets for middle corner in (x,y) unskewed coords
		y1 = y0 - j1 + G2
		x2 = x0 - 1.0 + 2.0 * G2  # Offsets for last corner in (x,y) unskewed coords
		y2 = y0 - 1.0 + 2.0 * G2

		# Work out the hashed gradient indices of the three simplex corners
		ii = int(i) & 255
		jj = int(j) & 255
		gi0 = [0]*3
		gi1 = [0]*3
		gi2 = [0]*3
		gi0[0] = permMat[0,ii+permMat[0,jj]] % 12
		gi1[0] = permMat[0,ii+i1+permMat[0,jj+j1]] % 12
		gi2[0] = permMat[0,ii+1+permMat[0,jj+1]] % 12

		gi0[1] = permMat[1,ii+permMat[1,jj]] % 12
		gi1[1] = permMat[1,ii+i1+permMat[1,jj+j1]] % 12
		gi2[1] = permMat[1,ii+1+permMat[1,jj+1]] % 12

		gi0[2] = permMat[2,ii+permMat[2,jj]] % 12
		gi1[2] = permMat[2,ii+i1+permMat[2,jj+j1]] % 12
		gi2[2] = permMat[2,ii+1+permMat[2,jj+1]] % 12

		# Calculate the contribution from the three corners
		t0 = 0.5 - x0*x0 - y0*y0
		if t0 < 0:
			pass
		else:
			t0 *= t0**3
			n0[0] = t0 * dot2d(simpNoise._grad3[gi0[0]], x0, y0)
			n0[1] = t0 * dot2d(simpNoise._grad3[gi0[1]], x0, y0)
			n0[2] = t0 * dot2d(simpNoise._grad3[gi0[2]], x0, y0)

		t1 = 0.5 - x1*x1 - y1*y1
		if t1 < 0:
			pass
		else:
			t1 *= t1**3
			n1[0] = t1 * dot2d(simpNoise._grad3[gi1[0]], x1, y1)
			n1[1] = t1 * dot2d(simpNoise._grad3[gi1[1]], x1, y1)
			n1[2] = t1 * dot2d(simpNoise._grad3[gi1[2]], x1, y1)

		t2 = 0.5 - x2*x2-y2*y2
		if t2 < 0:
			pass
		else:
			t2 *= t2**3
			n2[0] = t2 * dot2d(simpNoise._grad3[gi2[0]], x2, y2)
			n2[1] = t2 * dot2d(simpNoise._grad3[gi2[1]], x2, y2)
			n2[2] = t2 * dot2d(simpNoise._grad3[gi2[2]], x2, y2)

		# Add contributions from each corner to get the final noise value.
		# The result is scaled to return values in the interval [-1,1].
		return [70.0 * (n0[0] + n1[0] + n2[0]), 70.0 * (n0[1] + n1[1] + n2[1]), 70.0 * (n0[2] + n1[2] + n2[2]) ]


class nRand():
	def __init__(self, seed=0):
		self.rndState = None
		self.nprndState = None
		self.seed(int(seed))

	def seed( self, seedValue):
		#sys.stderr.write('seedValue '+str( seedValue  )+'\n' )
		random.seed(seedValue)
		self.rndState = random.getstate()
		self.nprndState = np.random.RandomState(seedValue)

	def resume( self ):
		random.setstate(self.rndState)

	def freeze( self ):
		self.rndState = random.getstate()

	def randInt( self, min, max ):
		return random.randint(min, max)

	def npaRandInt( self, min, max, sizeArray ):
		return self.nprndState.randint(min, max, size=sizeArray)

	def npaRand( self, min, max, sizeArray ):
		return self.nprndState.uniform(min,max,size=sizeArray)

# aAmplitudeRand is modified by this function
def npaRandSphereElevation( aElevation, aAmplitude, aPhi, aAmplitudeRand,  N ):
	sphPoints = np.empty( (N,3), np.float32 )

	theta = aAmplitudeRand
	theta *=aAmplitude
	theta -= aElevation
	theta -= aElevation
	theta += 1
	np.clip(theta,-1.0,1.0,out=theta)
	theta = np.arccos(theta)

	sphPoints[:,1] = sphPoints[:,2] = np.sin( theta )
	sphPoints[:,0] =  np.cos( theta )
	sphPoints[:,1] *= np.cos( aPhi )
	sphPoints[:,2] *= np.sin( aPhi )
	return sphPoints

# Global parameters
eGLOBAL = enum('time','detail','maxGeneration','tubeSides','seedChaos','seedSkeleton','doAccumulateTime','startTimeAccumulation',  'chaosSecondaryFreqFactor', 'vibrationFreqFactor', 'secondaryChaosMinClamp', 'secondaryChaosMaxClamp', 'secondaryChaosMinRemap', 'secondaryChaosMaxRemap', 'max')

# different types of values
	# default : On value for the overall path
	# AP : Along Path, it's an array descripbing the variation along the full path of a branch

# different scopes
	# default : the values apply to all the branches
	# BR : Per Branch, means each branch may have different values for this attributes
	# GEN : Per Generation, means all branch of the same generation share this value

# different transmission behavior
	# default : there is transfer value that is multiply to it before transmitting to next generation
	# SPE : the transmission to next generation is Special

# Along Path values Per Branch (attributes transmitted to children by mult)
eAPBR = enum('radius', 'intensity', 'length', 'max')

# Special Values Per Branch enum (no transmitted by function)
# seedSkeleton is the seed that drive the number of random childs and their position along the path
# seedChaos is the seed that drive the shape of the tube chaos
eSPEBR = enum('seedSkeleton', 'seedChaos', 'max')

# Values Depending on Generation and that are transmitted to the next generation by factor vector
eGEN = enum('childrenNumber', 'childrenNumberRand', 'skeletonTime', 'chaosTime', 'chaosDisplacement', 'chaosFrequency', 'chaosVibration', 'lengthRand', 'max')

# Along Path values Special ( No Transfert to Child )
eAPSPE = enum('elevation', 'elevationRand', 'childProbability', 'chaosDisplacement', 'max')

class lightningBoltProcessor:

	_frameTemplate = np.array( [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]], np.float32 )
	_childFrameCalcTemplate = np.array( [[0,0,0],[0,-1,0],[-1,0,0]], np.float32 ) # front, up, side (initial values are the singularity case of the algorithm "Building an Orthonormal Basis from a 3D Unit Vector Without Normalization" )

	def __init__( self, outSoftwareFunc = None ):
		self.outSoftwareFunc = outSoftwareFunc

		self.timer = avgTimer(30)

		self.maxPoints = 0

		self.randGeneralSeedChaos = nRand(0)
		self.randGeneralSeedSkeleton = nRand(0)

		self.previousTime = 0.0

		# APVariations are Along Path Variations (sampled curves), APVariations are the variations that are given to the system as inputs (they are maybe set by external ramps)
		# APVariations is a big numpy array where lines are the parameter along the path and column are alls the attributes
		# column 0 is Radius
		# column 1 is Intensity
		# column 2 is Offset
		self.APBR = None
		self.APBR_ROOT = None
		# this is a vector with all the multiplicator to apply to APVInputs
		self.APBRFactorsInit = np.zeros((eAPBR.max),np.float32)
		self.APBRTransfert = np.ones(eAPBR.max,np.float32) # factors for transfert of attribute from parent to child
		self.APBRTransfert_ROOT = np.ones(eAPBR.max,np.float32) # factors for transfert of attribute 


		# Per Branch Special values (is not transmitted by function to child)
		# 0 seedSkeleton (the seed that decided the number of childrens and the parameters of each one)
		self.SPEBRInit = np.zeros((eSPEBR.max),np.float32)

		# has no variation along the path is transmitted to child
		# O numchildens (nombre d'enfant genere obligatoirement)
		# 1 childrenNumberRand (nombre d'enfant potentiellement genere aleatoirement en plus)
		# 2 skeletonTimeMult (Multiplicator to timeSkeleton used to generate the childs number and params)
		self.GEN = np.zeros((eGEN.max),np.float32)
		self.GENTransfert = np.ones(eGEN.max,np.float32) # mult for transfert of attribute from parent to child
		self.GENTransfert_ROOT = np.ones(eGEN.max,np.float32) # mult for transfert of attribute from parent to child

		self.APSPE = None
		self.APSPE_ROOT = None

		self.startSeedPoints = []
		self.numStartSeedBranch = 0

		self.GLOBALS = [None]*eGLOBAL.max
		self.setGlobalValue( eGLOBAL.maxGeneration, 0 )
		self.setGlobalValue( eGLOBAL.tubeSides, 4 )
		self.setGlobalValue( eGLOBAL.seedChaos, 0 )
		self.setGlobalValue( eGLOBAL.seedSkeleton, 0 )
		self.setGlobalValue( eGLOBAL.chaosSecondaryFreqFactor, 2.0 )
		self.setGlobalValue( eGLOBAL.secondaryChaosMinClamp, 0.25 )
		self.setGlobalValue( eGLOBAL.secondaryChaosMaxClamp, 0.75 )
		self.setGlobalValue( eGLOBAL.secondaryChaosMinRemap, 0.5 )
		self.setGlobalValue( eGLOBAL.secondaryChaosMaxRemap, 1.0 )
		self.setGlobalValue( eGLOBAL.vibrationFreqFactor, 6.0 )
		self.setGlobalValue( eGLOBAL.doAccumulateTime, 0 )
		self.setGlobalValue( eGLOBAL.startTimeAccumulation, 1.0 )

# Global parameters
	def setGlobalValue( self, num, value):		
		if num == eGLOBAL.detail:
			if self.GLOBALS[eGLOBAL.detail] is not None and int(value) == self.GLOBALS[eGLOBAL.detail]:
				return
			self.GLOBALS[eGLOBAL.detail] = int(value)
			self.maxPoints = 2 + 2**self.GLOBALS[eGLOBAL.detail]
			self.APBR = np.zeros((self.maxPoints,eAPBR.max),np.float32)
			self.APBR_ROOT = np.zeros((self.maxPoints,eAPBR.max),np.float32)
			self.APSPE = np.zeros((self.maxPoints,eAPSPE.max),np.float32)
			self.APSPE_ROOT = np.zeros((self.maxPoints,eAPSPE.max),np.float32)
			self._seedPathStep = np.linspace(0.0,1.0,self.maxPoints)
			self._onesPathStep = np.ones(self.maxPoints,np.float32)
		elif num == eGLOBAL.tubeSides:
			if self.GLOBALS[eGLOBAL.tubeSides] is not None and int(value) == self.GLOBALS[eGLOBAL.tubeSides]:
				return
			self.GLOBALS[eGLOBAL.tubeSides] = int(value)
			# circle template for tube extrusion
			self.circleTemplate = np.zeros( (self.GLOBALS[eGLOBAL.tubeSides],4), np.float32 ) # (X=0,Y,Z,W)
			self.circleTemplate[:,1] = np.sin(np.linspace(1.5*np.pi,-np.pi*.5, self.GLOBALS[eGLOBAL.tubeSides], endpoint=False))
			self.circleTemplate[:,2] = np.cos(np.linspace(1.5*np.pi,-np.pi*.5, self.GLOBALS[eGLOBAL.tubeSides], endpoint=False))
			self.circleTemplate[:,3] = np.ones(self.GLOBALS[eGLOBAL.tubeSides])
			# intensity color template
			self.colorTemplate = np.ones( (self.GLOBALS[eGLOBAL.tubeSides],4), np.float32 )
		elif num == eGLOBAL.maxGeneration:
			if self.GLOBALS[eGLOBAL.maxGeneration] is not None and int(value) == self.GLOBALS[eGLOBAL.maxGeneration]:
				return
			self.GLOBALS[eGLOBAL.maxGeneration] = int(value)
			self.timesAccumulator = np.zeros((self.GLOBALS[eGLOBAL.maxGeneration]+1,2), np.float32)
		# we parse generic integer values
		elif num in [ eGLOBAL.seedChaos, eGLOBAL.seedSkeleton, eGLOBAL.doAccumulateTime ] :
			self.GLOBALS[num] = int(value)
		else: # others are generic float values
			self.GLOBALS[num] = float(value)

# Along Path Values Per Branch
	def getAPBR(self, num):
		return self.APBR[:,num], self.APBR_ROOT[:,num]

	def setAPVFactors(self, num, value):
		self.APBRFactorsInit[num] = value

	def setAPVTransfert(self, num, value, valueRoot=None):
		self.APBRTransfert[num] = value
		if valueRoot is None:
			self.APBRTransfert_ROOT[num] = value
		else:
			self.APBRTransfert_ROOT[num] = valueRoot

# Generation Dependent
	def setGENValue(self, num, value):
		self.GEN[num] = value
	def setGENTransfert(self, num, value, valueRoot=None):
		self.GENTransfert[num] = value
		if valueRoot is None:
			self.GENTransfert_ROOT[num] = value
		else:
			self.GENTransfert_ROOT[num] = valueRoot

# Along Path Special Values
	def getAPSPE(self, num):
		return self.APSPE[:,num], self.APSPE_ROOT[:,num]
		
# Specials Values Per Branch
	def setSpecialBranchValue(self, num, value):
		self.SPEBRInit[num] = value

#----------	
	def addSeedPointList( self, pointList):
		self.startSeedPoints.extend( pointList )
		self.numStartSeedBranch =  self.numStartSeedBranch + 1


	def generateInitialBatch( self ):
		onesPerBranch = np.ones(self.numStartSeedBranch,np.float32)

		self.randGeneralSeedChaos = nRand(self.GLOBALS[eGLOBAL.seedChaos])
		self.randGeneralSeedSkeleton = nRand(self.GLOBALS[eGLOBAL.seedSkeleton])

		# get seedPath directions and magnitude from seedPath points
		startSeedPath = np.array( self.startSeedPoints ).reshape(self.numStartSeedBranch,self.maxPoints,3)
		dirVectors = np.roll( startSeedPath, -1, axis=1 )
		unitDir = dirVectors[:,:-1]
		unitDir -= startSeedPath[:,:-1]
		normDir = np.sqrt(((unitDir**2).sum(-1)).reshape(self.numStartSeedBranch,self.maxPoints-1,1))
		unitDir /= normDir
		lengths = normDir.reshape(self.numStartSeedBranch,-1).sum(-1)

		# set APVMult (especially set the length)
		startSeedAPVMult = self.APBRFactorsInit*onesPerBranch[:,np.newaxis]
		startSeedAPVMult[:,eAPBR.length] *= lengths 

		# set time
		#self.GEN[eGEN.chaosTime] *= self.GLOBALS[eGLOBAL.time]
		#self.GEN[eGEN.skeletonTime] *= self.GLOBALS[eGLOBAL.time]

		# set SPE Values (especially set the seeds)
		startSeedBranchSpeValues = self.SPEBRInit*onesPerBranch[:,np.newaxis]
		startSeedBranchSpeValues[:,eSPEBR.seedSkeleton] = self.randGeneralSeedSkeleton.npaRandInt(0,1000000, self.numStartSeedBranch)
		startSeedBranchSpeValues[:,eSPEBR.seedChaos] = self.randGeneralSeedChaos.npaRandInt(0,1000000, self.numStartSeedBranch)

		return self.numStartSeedBranch, startSeedPath, dirVectors,  startSeedAPVMult, startSeedBranchSpeValues


	def generate( self, time, timesAccumulator, batch, APBranchV, APSpeV, APBranchTransfert, GenValues, GenTransfert, isLooping, doGenerateChilds ):
		epsilon = 0.00001

		# unpack
		batchSize, seedPathPoints, seedUnitDir, APBranchMults, SPEBRValues = batch
		#sys.stderr.write('APBranchMults '+str(APBranchMults)+'\n')
		#sys.stderr.write('seedUnitDir '+str(seedUnitDir)+'\n')
		#sys.stderr.write('APBranchV '+str(APBranchV)+'\n')
		#sys.stderr.write('batchSize '+str(batchSize)+'\n')
		#sys.stderr.write('Values '+str(Values)+'\n')
		#totalBatchPoints = len(seedPath) # this should be self.maxPoints*batchSize

		# get the final values at each path point ( the value from the ramp multiply by the factor )
		APBranchesArray = APBranchMults.reshape(batchSize,1,-1)*APBranchV

		timesAccumulator[0] += GenValues[eGEN.skeletonTime]*time
		timesAccumulator[1] += GenValues[eGEN.chaosTime]*time
	
		# For generating childs
		SPEBRValues[:,eSPEBR.seedSkeleton] += np.floor( timesAccumulator[0])# GenValues[eGEN.skeletonTime] )
		allChildsBranchParentId = []
		allRandValues = []

		chaosDisplacementArray = np.empty( (batchSize,self.maxPoints,3), np.float32 )
		freq = GenValues[eGEN.chaosFrequency]
		noisePos = np.empty( (self.maxPoints,2), np.float32 )
		noisePos[:,0] = timesAccumulator[1] #GenValues[eGEN.chaosTime]
		
		for i in range(batchSize):
			# we generate permutation tables for 5 dimensions (3 to output vector) (1 for amplitude) (1 for vibration)
			permTables = simpNoise.generatePermutationTables( SPEBRValues[i,eSPEBR.seedChaos], 5 )
			
			branchLength = APBranchMults[i,eAPBR.length]
			noisePos[:,1] = np.linspace(0.0,branchLength*freq,self.maxPoints)

		# primary noise ( perlin simplex output 3D Vector ) (time,branchParam)->(x,y,z)
			chaosDisplacementArray[i] = simpNoise.in2D_outVectDim(noisePos, permTables[:3] )

		# secondary noise ( perlin simplex output amplitude ) (time,branchParam)->w
			noisePos[:,1] *= self.GLOBALS[eGLOBAL.chaosSecondaryFreqFactor] 
			ampNoise = .5*simpNoise.in2D_outVectDim(noisePos, permTables[3:4] ) + .5
			np.clip(ampNoise, self.GLOBALS[eGLOBAL.secondaryChaosMinClamp], self.GLOBALS[eGLOBAL.secondaryChaosMaxClamp], out=ampNoise)
			ampNoise -= self.GLOBALS[eGLOBAL.secondaryChaosMinClamp]
			ampNoise *= ( self.GLOBALS[eGLOBAL.secondaryChaosMaxRemap] - self.GLOBALS[eGLOBAL.secondaryChaosMinRemap])/( self.GLOBALS[eGLOBAL.secondaryChaosMaxClamp] - self.GLOBALS[eGLOBAL.secondaryChaosMinClamp])
			ampNoise += self.GLOBALS[eGLOBAL.secondaryChaosMinRemap]
			#sys.stderr.write('ampNoise '+str(ampNoise)+'\n')

		# Vibration noise (pure random displacement) (time,branchParam)->v
			noisePos[:,1] *= self.GLOBALS[eGLOBAL.vibrationFreqFactor]
			vibrationsArray = GenValues[eGEN.chaosVibration]*simpNoise.in2D_outVectDim(noisePos, permTables[4:5] )
			
			# add vibration noise to amplitude Noise 
			ampNoise += vibrationsArray.reshape(-1,1)
			# multiply the amplitudeNoise to the vector from the 3D vector from the primary noise: Displacement = (w + v)*AmplitudeMax*(x,y,z)
			chaosDisplacementArray[i]*=ampNoise

		# childs calculations
			if not doGenerateChilds:
				continue

			randBranch = nRand(SPEBRValues[i,eSPEBR.seedSkeleton])
			branchChildNumber = int(GenValues[eGEN.childrenNumber]) + randBranch.randInt(0, int(GenValues[eGEN.childrenNumberRand] ) )
			randArray = randBranch.npaRand(epsilon, 1.0-epsilon, (branchChildNumber, 6) )
			allRandValues.extend( randArray.tolist() )
			allChildsBranchParentId.extend( [i]*branchChildNumber )
		
		# add the displacement offset to the seedPath point
		seedPathPoints += GenValues[eGEN.chaosDisplacement]*chaosDisplacementArray*APSpeV[:,eAPSPE.chaosDisplacement].reshape(1,-1,1)
		#sys.stderr.write('seedPath '+str(seedPath)+'\n')

	# from here we compute the frame ( unit front, unit up, unit side ) at each point
	# the final frame is a full 4,4 matrix, like this :
	#  fx ux sx px 
	#  fy uy sy py
	#  fz uz sz pz
	#  0  0  0  1
	#  where front is (fx,fy,fz,0)
	#  where side is (sx,sy,sz,0)
	#  where up is (ux,uy,uz,0)
	#  where path point is (px,py,pz,1)
	# this way we have a full transformation matrix with a good translation that we can use to get all the tube points.
	# algorithm is this:
	# - compute vector between all the points Di
	# - normalize Di
	# - unit front = Di-1 + Di (at each point the frame is "tangent")
	# - normalize unit front and unit front is then ok
	# - complete for every branch the initial frame. Get a unit side from an arbitrary cross(front,[0,1,0]) or cross(front, [1,0,0]) if [0,1,0] and front are parallel. Then get the unit Up by cross( side, front)
	# - then apply the algorithm (Double Reflection) for each branch ( http://research.microsoft.com/en-us/um/people/yangliu/publication/computation%20of%20rotation%20minimizing%20frames.pdf ) . This algorithm compute a good frame from a previous frame, so we iteratively compute all the frames along the paths. 

		# let's create the frame for every points on every branch
		frames = np.tile(lightningBoltProcessor._frameTemplate,(batchSize,self.maxPoints,1,1))

		frameFronts = frames[:,:,:-1,0] # get a view for the front only
		
		# compute vector between all the points Di = roll( pathPoint, -1 ) - pathPoint
		dirVectors = np.roll( seedPathPoints, -1, axis=1 )
		unitDir = dirVectors[:,:-1]
		unitDir -= seedPathPoints[:,:-1]
		normFrontSqr = np.sqrt(((unitDir**2).sum(-1)).reshape(batchSize,self.maxPoints-1,1))
		unitDir /= normFrontSqr

		# front = D(i-1) + D(i)
		#frameFronts[:,:-1,:][...] = unitDir
		frameFronts[:,:-1,:] = unitDir
		frameFronts[:,1:,:] += unitDir

		# normalize front (different case if we loop (the first point of the branch is different when we loop) )
		if isLooping :
			frameFronts[:,0,:] += frameFronts[:,-1,:] # we add the last direction of the branch to the first direction
			# we normalize all the direction except the last one ( -> :-1 )
			normFrontSqr = np.sqrt(((frameFronts[:,:-1,:]**2).sum(-1)).reshape(batchSize,self.maxPoints-1,1))
			frameFronts[:,:-1,:] /= normFrontSqr
		else : # we normalize all the direction except the first one and the last one ( -> 1:-1 )
			normFrontSqr = np.sqrt(((frameFronts[:,1:-1,:]**2).sum(-1)).reshape(batchSize,self.maxPoints-2,1))
			frameFronts[:,1:-1,:] /= normFrontSqr

		#complete for every branch the initial frame
		firstFrameViewFront = frames[:,0,:-1,0]
		firstFrameViewUp = frames[:,0,:-1,1]
		firstFrameViewSide = frames[:,0,:-1,2]

		firstFrameViewUp[:] = [0,1,0] # we initialize Up with vector Y
		firstFrameViewFrontY = frames[:,0,1,0]
		#sys.stderr.write('orig where '+str(np.where( np.abs( (firstFrameViewUp*firstFrameViewFront).sum(-1)) > 0.99999 ))+'\n')
		#sys.stderr.write('new where '+str( np.where( np.abs(firstFrameViewFrontY)>0.99999 )  )+'\n')
		verticalStartBranch = np.where( np.abs(firstFrameViewFrontY)>0.99999 )
		firstFrameViewUp[verticalStartBranch] = [1,0,0] # where it's vertical let's start with this vector instead

		# with a valid Up we can deduce the side then the up again to have a proper frame
		firstFrameViewSide[:] = np.cross( firstFrameViewFront, firstFrameViewUp )
		firstFrameViewSide /= np.sqrt( ((firstFrameViewSide**2).sum(-1)).reshape(-1,1) )

		firstFrameViewUp[:] = np.cross( firstFrameViewSide, firstFrameViewFront )

		# now apply the algorithm (Double Reflection) for each branch
		for i in range(1,self.maxPoints):
			frontPrevPoint = frames[:,i-1,:-1,0]
			upPrevPoint = frames[:,i-1,:-1,1]

			frontCurrentPoint = frames[:,i,:-1,0]
			upCurrentPoint = frames[:,i,:-1,1]
			sideCurrentPoint = frames[:,i,:-1,2]

			upL = ((unitDir[:,i-1,:]*upPrevPoint).sum(-1)*-2.0).reshape(-1,1)*unitDir[:,i-1,:]
			upL += upPrevPoint
			frL = ((unitDir[:,i-1,:]*frontPrevPoint).sum(-1)*2.0).reshape(-1,1)*unitDir[:,i-1,:]
			frL -= frontPrevPoint
			v2 = frL # make v2 the same object as frL (frL is used only for v2)
			v2 += frontCurrentPoint
			c2 = (v2*v2).sum(-1).reshape(-1,1) # dot

			upCurrentPoint[...] = ((v2*upL).sum(-1)*-2.0).reshape(-1,1)*v2
			upCurrentPoint /= c2
			upCurrentPoint += upL
			sideCurrentPoint[...] = np.cross(frontCurrentPoint,upCurrentPoint)

		# multiply frames by corresponding Radius ( we scale only the vectors )
		frames[:,:,:-1,:-1] *= APBranchesArray[:,:,eAPBR.radius,np.newaxis,np.newaxis]

		# load branch points in frames to complete the transformation matrix (we are done with the frames!)
		frames[:,:,:-1,3] = seedPathPoints

	# now we can generate childs if we need
		childsBatch = None
		totalChildNumber = len(allChildsBranchParentId)			
		if totalChildNumber>0 :
			aChildRandomsArray = np.array( allRandValues, np.float32 )
			aChildRandomsArray = aChildRandomsArray.T

			#sys.stderr.write('aChildRandomsArray[0,:] '+str(aChildRandomsArray[0,:])+'\n')

			aChildRandomsArray[0,:]*=float(self.maxPoints-1)# bring the parameters back to [0,maxPoints-1]
			#sys.stderr.write('aChildRandomsArray[0,:] '+str(aChildRandomsArray[0,:])+'\n')
			indexT0 = aChildRandomsArray[0,:].astype(np.uint32)
			blend = aChildRandomsArray[0,:] # VIEW
			blend -= indexT0
			blend = blend.reshape(-1,1)
			
			#sys.stderr.write('indexT0 '+str(indexT0)+'\n')
			indexT0 = ((float(self.maxPoints-1)-epsilon)*APSpeV[indexT0,eAPSPE.childProbability]  ).astype(np.uint32)

		# Get the Along Path Per Generation Values for the elevation and elevationRand for all theses childs
			#sys.stderr.write('APSpeV elevation '+str(APSpeV[:,eAPSPE.elevation])+'\n')
			APGenT0 = APSpeV[indexT0,eAPSPE.elevation:eAPSPE.elevationRand+1]
			APGenChildElevationAndElevationRand =  APGenT0 + blend*( APSpeV[indexT0+1,eAPSPE.elevation:eAPSPE.elevationRand+1] - APGenT0 )

		# Now get the Along Path Per Branch Values
			aChildParentId = np.array(allChildsBranchParentId, np.float32)
			indexT0 += aChildParentId*self.maxPoints # indices must now point to the correct parent branch
			#sys.stderr.write('indexT0 '+str(indexT0)+'\n')

			#sys.stderr.write('aChildRandomsArray[0] '+str(aChildsParamGlobal)+'\n')
			#sys.stderr.write('allChildsBranchParentId '+str(allChildsBranchParentId)+'\n')

			APVView = APBranchesArray.reshape(-1,eAPBR.max)
			APVViewT0 = APVView[indexT0]
			APBranchMultsChilds = APVViewT0 + blend*( APVView[indexT0+1] - APVViewT0 )
			APBranchMultsChilds *= APBranchTransfert #self.APBRTransfert # transfert Along Path Branch multiply

		# Then the Special Values for child branches
			SPEBRValuesChilds = np.empty( (totalChildNumber,eSPEBR.max ), np.float32  )
			#SPEBRValuesChilds = SPEBRValues[allChildsBranchParentId]

			# set the seeds of the childs
			aAllSeeds = aChildRandomsArray[3:5,:]
			aAllSeeds *= 1000000.0
			aAllSeeds %= 1000000.0
			SPEBRValuesChilds[:,eSPEBR.seedSkeleton] = aAllSeeds[0]
			SPEBRValuesChilds[:,eSPEBR.seedChaos] = aAllSeeds[1]
		
		# now create frames for segments ( based on dirVectors ) so we can control the direction for the child branch
			# multiple childs can be generated on the same segment, we don't want to calculate extras frame system
			paramIdToCalc = np.unique(indexT0)
			#sys.stderr.write('paramIdToCalc'+str(paramIdToCalc)+'\n')
			reverseToChildId = np.searchsorted( paramIdToCalc, indexT0.flat) # this could be removed if numpy.unique has the reverse_index parameter 

			# from the algorithm of Jeppe Revall Frisvad "Building an Orthonormal Basis from a 3D Unit Vector Without Normalization"
			childFramesCalc = np.tile(lightningBoltProcessor._childFrameCalcTemplate,(len(paramIdToCalc),1,1))
			# we use the seed directions of segments (seedUnitDir), this way the elevation direction of childs is independant from the chaos
			childFramesCalc[:,0] = seedUnitDir.reshape(-1,3)[paramIdToCalc]

			okIds = np.where( childFramesCalc[:,0,2]>-1.0 )
			#sys.stderr.write('okIds'+str(okIds)+'\n')

			okIdsFronts = childFramesCalc[okIds[0],0,:]
			#sys.stderr.write('okIdsFronts '+str(okIdsFronts)+'\n')
			#sys.stderr.write('okIdsFrontsZ '+str(okIdsFronts[:,2])+'\n')
			#sys.stderr.write('OLD '+str(childFramesCalc[okIds[0],0,2])+'\n')
			
			vala = 1.0/(1.0 + okIdsFronts[:,2] )
			valb = -okIdsFronts[:,0]*okIdsFronts[:,1]*vala
			# up
			childFramesCalc[okIds[0],1,0] = 1.0 - vala*(okIdsFronts[:,0]**2)
			childFramesCalc[okIds[0],1,1] = valb
			childFramesCalc[okIds[0],1,2] = -okIdsFronts[:,0]
			# side
			childFramesCalc[okIds[0],2,0] = valb
			childFramesCalc[okIds[0],2,1] = 1.0 - vala*(okIdsFronts[:,1]**2)
			childFramesCalc[okIds[0],2,2] = -okIdsFronts[:,1]

			# here we have the full array for every point
			childFrames = childFramesCalc[reverseToChildId]
		# now get the elevation for each child branch
			aChildRandomsArray[1:3,:]*=2.0
			aPhi = aChildRandomsArray[1,:] #aPhiAndRandAmplitude[0]
			aAmplitudeRand = aChildRandomsArray[2,:] #aPhiAndRandAmplitude[1]
			aPhi*=np.pi 
			aAmplitudeRand-=1.0
			 
			randomVectors = npaRandSphereElevation( APGenChildElevationAndElevationRand[:,0], APGenChildElevationAndElevationRand[:,1], aPhi, aAmplitudeRand, aAmplitudeRand.size  )

			#sys.stderr.write('childFrames'+str(childFrames)+'\n')
			#sys.stderr.write('randomVectors'+str(randomVectors)+'\n')
			childFrames*=randomVectors.reshape(-1,3,1) # here we use the random vector coordinate expressed in the frame of each segments
			#sys.stderr.write('childFrames'+str(childFrames)+'\n')
			childFrames = childFrames.transpose(0,2,1).sum(-1) # we get only the front vector
			#sys.stderr.write('childFrames'+str(childFrames)+'\n')
			
			unitDirChilds = childFrames[:,np.newaxis]*self._onesPathStep[:,np.newaxis] # we keep an array of dimension numPoint despite the fact that this is direction and there is one direction less than the number point because it simplify the calculus of indices, indices are the same for direction array and for points
			#sys.stderr.write('unitDirChilds'+str(unitDirChilds) +'\n' )
			# the forumalae for length is length*(1.0 + lengthRand*Rand(0,1))
			aChildRandomsArray[5,:] *= GenValues[eGEN.lengthRand] #lengthRand*Rand(0,1)
			aChildRandomsArray[5,:] += 1.0 # lengthRand*Rand(0,1)+1
			childFrames *= aChildRandomsArray[5,:,np.newaxis]*APBranchMultsChilds[:,eAPBR.length,np.newaxis] # length*(1.0 + lengthRand*Rand(0,1))

			#sys.stderr.write('child length tot'+str(aChildRandomsArray[5,:,np.newaxis]*APBranchMultsChilds[:,eAPBR.length,np.newaxis]) +'\n' )
			seedPathChilds = childFrames[:,np.newaxis]*self._seedPathStep[:,np.newaxis]
			#sys.stderr.write('seedPathChilds'+str(seedPathChilds) +'\n' )

			seedPathViewLine = seedPathPoints.reshape(-1,3)
			seedPathT0 = seedPathViewLine[indexT0]
			startPoints = seedPathT0 + blend*( seedPathViewLine[indexT0+1] - seedPathT0 )

			#sys.stderr.write('startPoints'+str(startPoints) +'\n' )
			#tiled = np.tile(startPoints.T,(self.maxPoints,1) )
			#sys.stderr.write('startPoints'+str(startPoints) +'\n' )
			#sys.stderr.write('startPoints[:,np.newaxis]*onesPathStep[:,np.newaxis]'+str(startPoints[:,np.newaxis]*onesPathStep[:,np.newaxis]) +'\n' )
			#sys.stderr.write('tiled'+str(tiled.reshape(totalChildNumber,3,-1) ) +'\n' )

			seedPathChilds += startPoints[:,np.newaxis]*self._onesPathStep[:,np.newaxis]

			# Apply the transfert to the Generation Values
			GenValues *= GenTransfert # transfert Values multiply

			#sys.stderr.write('seedPathChilds1'+str(seedPathChilds[1]) +'\n' )
			#sys.stderr.write('APBranchMultsChilds'+str(APBranchMultsChilds) +'\n' )
			childsBatch = ( totalChildNumber, seedPathChilds , unitDirChilds,  APBranchMultsChilds, SPEBRValuesChilds )
			#batchSize, seedPath, APVMults, Values, GValues = batch
			
			#sys.stderr.write('APArray[0,:,:,lightningBoltProcessor.eAPV.intensity]'+str(APArray[0,:,:,lightningBoltProcessor.eAPV.intensity]) +'\n' )

		return frames, APBranchesArray[:,:,eAPBR.intensity], childsBatch

	def process(self ):		
		currentGeneration = 0

		# --- get the first datas (seedPath, attributes, etc... ) to process 
		batch = self.generateInitialBatch()
		
		APBranch = self.APBR_ROOT
		APBranchTransfert = self.APBRTransfert_ROOT
		APSpe = self.APSPE_ROOT
		GenTransfert = self.GENTransfert_ROOT
		
		GenValues =  self.GEN.copy() # Per generation Values are modified by a transfert

		self.timer.start()

		# The result returned will be points in the form of a huge list of coordinate, there will be 2 lists to separate ring and non ring lightning
		resultLoopingFrames = []
		numLoopingLightningBranch = 0
		resultFrames = []
		numLightningBranch = 0
		isLooping = False
		resultIntensities = []

		doAccum = False
		startTime = 10.0

		time = self.GLOBALS[eGLOBAL.time]
		if doAccum:
			if time<startTime+0.000001:
				self.timesAccumulator[...] = 0.0
				self.previousTime = time
			time -= self.previousTime # we send a delta
		else:
			self.timesAccumulator[...] = 0.0
		self.previousTime = self.GLOBALS[eGLOBAL.time]

		while batch is not None:
			outFrames, outIntensities, childBatch = self.generate( time, self.timesAccumulator[currentGeneration,:], batch, APBranch, APSpe, APBranchTransfert, GenValues, GenTransfert, isLooping, currentGeneration<self.GLOBALS[eGLOBAL.maxGeneration])

			if isLooping :
				resultLoopingFrames.append(outFrames.reshape(-1,4,4))
				numLoopingLightningBranch += batch[0]
			else:
				resultFrames.append(outFrames.reshape(-1,4,4))
				numLightningBranch += batch[0]
				resultIntensities.append(outIntensities.reshape(-1))

			batch = childBatch
			# set the generation to the general values
			APBranch = self.APBR
			APBranchTransfert = self.APBRTransfert
			APSpe = self.APSPE
			GenTransfert = self.GENTransfert

			isLooping = False # only the first generation can loop
			currentGeneration = currentGeneration + 1

		# here deduce points from frame and circles and add them to resultPoints
		# transform a circle of point for each frame
		resultPoints = [ coord for array in resultFrames for coord in ((np.inner(array,self.circleTemplate)).transpose(0,2,1)).reshape(-1).tolist() ]
		resultIntensityColors = [ coord for array in resultIntensities for coord in (self.colorTemplate*array[:,np.newaxis,np.newaxis]).reshape(-1).tolist() ]
		#sys.stderr.write('resultPoints '+str(resultPoints)+'\n')

		resultLoopingPoints = [] # TODO
		numLoopingLightningBranch = 0 # TODO

		average = self.timer.stop()
		#sys.stderr.write('time '+str(average)+'\n')

		# We reset the processor before leaving
		self.startSeedPoints = []
		self.numStartSeedBranch = 0

		# then we call the external (software-dependant meshing)
		return self.outSoftwareFunc( resultLoopingPoints, numLoopingLightningBranch, resultPoints, numLightningBranch, resultIntensityColors, self.maxPoints, self.GLOBALS[eGLOBAL.tubeSides] )

