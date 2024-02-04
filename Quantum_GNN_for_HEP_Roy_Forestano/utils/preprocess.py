import numpy as np
import torch
from torch import nn
from particle import Particle
from tqdm import tqdm
import copy
import pennylane as qml

'''
------------------------------------------------------------------------------------------
 Euclidean distance
------------------------------------------------------------------------------------------
'''
def get_distance(tuple1,tuple2):
    return np.sqrt((tuple1[0]-tuple2[0])**2+(tuple1[1]-tuple2[1])**2)


'''
------------------------------------------------------------------------------------------
 Using graphs with a general number of nodes
------------------------------------------------------------------------------------------
'''
def preprocess_general(x_data,y_data): #,masses):
    print('--- Finding All Unique Particles ---')
    unique_particles = np.unique(x_data[:,:,3])
    x_data = torch.tensor(x_data)
    y_data = torch.tensor(y_data)
    print()
    print('--- Inserting Masses ---')
    masses = torch.zeros((x_data.shape[0],x_data.shape[1]))
    for i,particle in tqdm(enumerate(unique_particles)):
        if particle!=0:
            mass = Particle.from_pdgid(particle).mass/1000
            inds = torch.where(particle==x_data[:,:,3])
            masses[inds]=mass # GeV
    print()
    print('--- Calculating Momenta and Energies ---')
    #theta = torch.arctan(torch.exp(-X[:,:,1]))*2 # polar angle
    pt        = x_data[:,:,0]     # transverse momentum
    rapidity  = x_data[:,:,1]     # rapidity
    phi       = x_data[:,:,2]     # azimuthal angle
    
    mt        = (pt**2+masses**2).sqrt() # Transverse mass
    energy    = mt*torch.cosh(rapidity) # Energy per multiplicity bin
    e_per_jet = energy.sum(axis=1)  # total energy per jet summed across multiplicity bins

    px = pt*torch.cos(phi)  # momentum in x
    py = pt*torch.sin(phi)  # momentum in y
    pz = mt*torch.sinh(rapidity)  # momentum in z
    
    # three momentum
    p  = torch.cat(( px[:,:,None],  
                     py[:,:,None],
                     pz[:,:,None]), dim=2 )

    p_per_jet        = (p).sum(axis=1)  # total componet momentum per jet
    pt_per_Mbin      = (p_per_jet[:,:2]**2).sum(axis=1).sqrt()  # transverse momentum per jet
    mass_per_jet     = (e_per_jet**2-(p_per_jet**2).sum(axis=1)).sqrt() # mass per jet
    rapidity_per_jet = torch.log( (e_per_jet+p_per_jet[:,2])/(e_per_jet-p_per_jet[:,2]) )/2  # rapidity per jet from analytical formula
    end_multiplicity_indx_per_jet = (pt!=0).sum(axis=1).int() # see where the jet (graph) ends
    
    x_data = torch.cat( ( x_data[:,:,:3],
                          x_data[:,:,4:],
                          masses[:,:,None],
                          energy[:,:,None],
                          p), dim=2)
    
    x_data_max = (x_data.max(dim=1).values).max(dim=0).values
    x_data = x_data/x_data_max

    print()
    print('--- Calculating Edge Tensors ---')
    N = x_data[:,0,3].shape[0]  # number of jets (graphs)
    M = x_data[0,:,3].shape[0]  # number of max multiplicty
    connections = 3
    edge_tensor = torch.zeros((N,M,M))
    edge_indx_tensor = torch.zeros((N,2,M*(connections+1))) # M*(connections+1) is the max number of edges we allow per jet
    edge_attr_matrix = torch.zeros((N,M*(connections+1),1)) 
    for jet in tqdm(range(N)):
        stop_indx = end_multiplicity_indx_per_jet[jet] # stop finding edges once we hit zeros
        if end_multiplicity_indx_per_jet[jet]>connections:
            for m in range(end_multiplicity_indx_per_jet[jet]):
                inds_edge = np.argsort((energy[jet,m]+energy[jet,:stop_indx])**2-torch.sum((p[jet,m,:stop_indx]+p[jet,:stop_indx,:])**2,axis=1))[:connections]
                edge_tensor[jet,m,inds_edge] = (energy[jet,m]+energy[jet,inds_edge])**2-torch.sum((p[jet,m,:]+p[jet,inds_edge,:])**2,axis=1)
                edge_tensor[jet,m,m]=((energy[jet,m]+energy[jet,m])**2-torch.sum((p[jet,m,:]+p[jet,m,:])**2,axis=0))
            edges_exist_at = torch.where(edge_tensor[jet,:,:].abs()>0)
            edge_indx_tensor[jet,:,:(edge_tensor[jet,:,:].abs()>0).sum()] = torch.cat((edges_exist_at[0][None,:],edges_exist_at[1][None,:]),dim=0).reshape((2,edges_exist_at[0].shape[0]))   
            edge_attr_matrix[jet,:(edge_tensor[jet,:,:].abs()>0).sum(),0]  =  edge_tensor[jet,edges_exist_at[0],edges_exist_at[1]].flatten()

    end_edges_indx_per_jet = (edge_attr_matrix!=0).sum(axis=1).int()
    keep_inds = np.where(end_edges_indx_per_jet!=0)[0]
    
    edge_tensor = edge_tensor/edge_tensor.max()
    edge_attr_matrix = edge_attr_matrix/edge_attr_matrix.max()
    

    graph_help = torch.cat( ( (energy.max(axis=1).values/e_per_jet).reshape(x_data[:,0,3].shape[0],1),
                              (mass_per_jet).reshape(x_data[:,0,3].shape[0],1),
                              (end_multiplicity_indx_per_jet).reshape(x_data[:,0,3].shape[0],1).int(),
                              (end_edges_indx_per_jet).reshape(x_data[:,0,3].shape[0],1).int() ), dim=1)
        
    return x_data[keep_inds], y_data[keep_inds].long(), edge_tensor[keep_inds], edge_indx_tensor[keep_inds].long(), edge_attr_matrix[keep_inds], graph_help[keep_inds]



'''
------------------------------------------------------------------------------------------
 Using graphs with a fixed number of nodes
------------------------------------------------------------------------------------------
'''
def preprocess_fixed_nodes(x_data,y_data,nodes_per_graph=10): #,masses):
    print('--- Finding All Unique Particles ---')
    unique_particles = np.unique(x_data[:,:,3])
    x_data = torch.tensor(x_data)
    y_data = torch.tensor(y_data)
    print()
    print('--- Inserting Masses ---')
    masses = torch.zeros((x_data.shape[0],x_data.shape[1]))
    for i,particle in tqdm(enumerate(unique_particles)):
        if particle!=0:
            mass = Particle.from_pdgid(particle).mass/1000
            inds = torch.where(particle==x_data[:,:,3])
            masses[inds]=mass # GeV
    print()
    print('--- Calculating Momenta and Energies ---')
    #theta = torch.arctan(torch.exp(-X[:,:,1]))*2 # polar angle
    pt        = x_data[:,:,0]     # transverse momentum
    rapidity  = x_data[:,:,1]     # rapidity
    phi       = x_data[:,:,2]     # azimuthal angle
    
    mt        = (pt**2+masses**2).sqrt() # Transverse mass
    energy    = mt*torch.cosh(rapidity) # Energy per multiplicity bin
    e_per_jet = energy.sum(axis=1)  # total energy per jet summed across multiplicity bins

    px = pt*torch.cos(phi)  # momentum in x
    py = pt*torch.sin(phi)  # momentum in y
    pz = mt*torch.sinh(rapidity)  # momentum in z
    
    # three momentum
    p  = torch.cat(( px[:,:,None],  
                     py[:,:,None],
                     pz[:,:,None]), dim=2 )

    p_per_jet        = (p).sum(axis=1)  # total componet momentum per jet
    pt_per_Mbin      = (p_per_jet[:,:2]**2).sum(axis=1).sqrt()  # transverse momentum per jet
    mass_per_jet     = (e_per_jet**2-(p_per_jet**2).sum(axis=1)).sqrt() # mass per jet
    rapidity_per_jet = torch.log( (e_per_jet+p_per_jet[:,2])/(e_per_jet-p_per_jet[:,2]) )/2  # rapidity per jet from analytical formula
    end_multiplicity_indx_per_jet = (pt!=0).sum(axis=1).int() # see where the jet (graph) ends
    
    x_data = torch.cat( ( x_data[:,:,:3],
                          x_data[:,:,4:],
                          masses[:,:,None],
                          energy[:,:,None],
                          p), dim=2)
    
    x_data_max = (x_data.max(dim=1).values).max(dim=0).values
    x_data = x_data/x_data_max

    print()
    print('--- Calculating Edge Tensors ---')
    N = x_data[:,0,3].shape[0]  # number of jets (graphs)
    M = nodes_per_graph #x_data[0,:,3].shape[0]  # number of max multiplicty
    connections = nodes_per_graph
    edge_tensor = torch.zeros((N,M,M))
    edge_indx_tensor = torch.zeros((N,2,connections*(connections-1) )) # M*(connections-1) is the max number of edges we allow per jet
    edge_attr_matrix = torch.zeros((N,connections*(connections-1),1)) 
#     fixed_edges_list = torch.tensor([ [i,j] for i in range(connections) for j in range(connections) if i!=j]).reshape(2,90)
    
    for jet in tqdm(range(N)):
        stop_indx = end_multiplicity_indx_per_jet[jet] #connections # stop finding edges once we hit zeros -> when we hit 10
        if end_multiplicity_indx_per_jet[jet]>=connections:
            for m in range(connections):
#                 inds_edge = np.argsort((energy[jet,m]+energy[jet,:stop_indx])**2-torch.sum((p[jet,m,:stop_indx]+p[jet,:stop_indx,:])**2,axis=1))[:connections]
#                 edge_tensor[jet,m,:] = (energy[jet,m]+energy[jet,:connections])**2-torch.sum((p[jet,m,:]+p[jet,:connections,:])**2,axis=1)
#                 edge_tensor[jet,m,m] = 0.
#                 edge_tensor[jet,m,m]=((energy[jet,m]+energy[jet,m])**2-torch.sum((p[jet,m,:]+p[jet,m,:])**2,axis=0))
                # inds_edge = torch.sqrt( (phi[jet,m]-phi[jet,:])**2 + (rapidity[jet,m]-rapidity[jet,:])**2 ).argsort()[:connections]
                # edge_tensor[jet,m,:] = torch.sqrt( (phi[jet,m]-phi[jet,inds_edge])**2 + (rapidity[jet,m]-rapidity[jet,inds_edge])**2 )
                edge_tensor[jet,m,:] = torch.sqrt( (phi[jet,m]-phi[jet,:connections])**2 + (rapidity[jet,m]-rapidity[jet,:connections])**2 )
#                 inds_edge = np.argsort( (energy[jet,m]+energy[jet,:stop_indx])**2-torch.sum((p[jet,m,:stop_indx]+p[jet,:stop_indx,:])**2,axis=1) )[:connections]
#                 edge_tensor[jet,m,inds_edge] = (energy[jet,m]+energy[jet,inds_edge])**2-torch.sum((p[jet,m,:]+p[jet,inds_edge,:])**2,axis=1)
            edges_exist_at = torch.where(edge_tensor[jet,:,:].abs()>0)
            
#             edge_indx_tensor[jet,:,:(edge_tensor[jet,:,:].abs()>0).sum()] = fixed_edges_list 
            edge_indx_tensor[jet,:,:(edge_tensor[jet,:,:].abs()>0).sum()] = torch.cat((edges_exist_at[0][None,:],edges_exist_at[1][None,:]),dim=0).reshape((2,edges_exist_at[0].shape[0]))   
            edge_attr_matrix[jet,:(edge_tensor[jet,:,:].abs()>0).sum(),0]  =  edge_tensor[jet,edges_exist_at[0],edges_exist_at[1]].flatten()

    end_edges_indx_per_jet = (edge_attr_matrix!=0).sum(axis=1).int()
    keep_inds =  torch.where(end_edges_indx_per_jet>=connections)[0]
    
    edge_tensor = edge_tensor/edge_tensor.max()
    edge_attr_matrix = edge_attr_matrix/edge_attr_matrix.max()
    
    graph_help = torch.cat( ( (energy.max(axis=1).values/e_per_jet).reshape(x_data[:,0,3].shape[0],1),
                              (mass_per_jet).reshape(x_data[:,0,3].shape[0],1),
                              (end_multiplicity_indx_per_jet).reshape(x_data[:,0,3].shape[0],1).int(),
                              (end_edges_indx_per_jet).reshape(x_data[:,0,3].shape[0],1).int() ), dim=1)
        
    return x_data[keep_inds,:nodes_per_graph], y_data[keep_inds].long(), edge_tensor[keep_inds], edge_indx_tensor[keep_inds].long(), edge_attr_matrix[keep_inds], graph_help[keep_inds]











'''
------------------------------------------------------------------------------------------
 Using graphs with a fixed number of nodes (QUANTUM)
------------------------------------------------------------------------------------------
'''
def preprocess_fixed_nodes_quantum(x_data,y_data,nodes_per_graph=10): #,masses):
    print('--- Finding All Unique Particles ---')
    unique_particles = np.unique(x_data[:,:,3])
    x_data = torch.tensor(x_data)
    y_data = torch.tensor(y_data)
    print()
    print('--- Inserting Masses ---')
    masses = torch.zeros((x_data.shape[0],x_data.shape[1]))
    for i,particle in tqdm(enumerate(unique_particles)):
        if particle!=0:
            mass = Particle.from_pdgid(particle).mass/1000
            inds = torch.where(particle==x_data[:,:,3])
            masses[inds]=mass # GeV
    print()
    print('--- Calculating Momenta and Energies ---')
    #theta = torch.arctan(torch.exp(-X[:,:,1]))*2 # polar angle
    pt        = x_data[:,:,0]     # transverse momentum
    rapidity  = x_data[:,:,1]     # rapidity
    phi       = x_data[:,:,2]     # azimuthal angle
    
    mt        = (pt**2+masses**2).sqrt() # Transverse mass
    energy    = mt*torch.cosh(rapidity) # Energy per multiplicity bin
    e_per_jet = energy.sum(axis=1)  # total energy per jet summed across multiplicity bins

    px = pt*torch.cos(phi)  # momentum in x
    py = pt*torch.sin(phi)  # momentum in y
    pz = mt*torch.sinh(rapidity)  # momentum in z
    
    # three momentum
    p  = torch.cat(( px[:,:,None],  
                     py[:,:,None],
                     pz[:,:,None]), dim=2 )

    p_per_jet        = (p).sum(axis=1)  # total componet momentum per jet
    pt_per_Mbin      = (p_per_jet[:,:2]**2).sum(axis=1).sqrt()  # transverse momentum per jet
    mass_per_jet     = (e_per_jet**2-(p_per_jet**2).sum(axis=1)).sqrt() # mass per jet
    rapidity_per_jet = torch.log( (e_per_jet+p_per_jet[:,2])/(e_per_jet-p_per_jet[:,2]) )/2  # rapidity per jet from analytical formula
    end_multiplicity_indx_per_jet = (pt!=0).sum(axis=1).int() # see where the jet (graph) ends
    
    x_data = torch.cat( ( x_data[:,:,:3],
                          x_data[:,:,4:],
                          masses[:,:,None],
                          energy[:,:,None],
                          p), dim=2)
    
    x_data_max = (x_data.max(dim=1).values).max(dim=0).values
    x_data = x_data/x_data_max

    print()
    print('--- Calculating Edge Tensors ---')
    N = x_data[:,0,3].shape[0]  # number of jets (graphs)
    M = nodes_per_graph #x_data[0,:,3].shape[0]  # number of max multiplicty
    connections = nodes_per_graph
    edge_tensor = torch.zeros((N,M,M))
    edge_indx_tensor = torch.zeros((N,2,connections*(connections-1) )) # M*(connections-1) is the max number of edges we allow per jet
    edge_attr_matrix = torch.zeros((N,connections*(connections-1),1)) 
#     fixed_edges_list = torch.tensor([ [i,j] for i in range(connections) for j in range(connections) if i!=j]).reshape(2,90)
    
    for jet in tqdm(range(N)):
        stop_indx = end_multiplicity_indx_per_jet[jet] #connections # stop finding edges once we hit zeros -> when we hit 10
        if end_multiplicity_indx_per_jet[jet]>=connections:
            for m in range(connections):
#                 inds_edge = np.argsort((energy[jet,m]+energy[jet,:stop_indx])**2-torch.sum((p[jet,m,:stop_indx]+p[jet,:stop_indx,:])**2,axis=1))[:connections]
#                 edge_tensor[jet,m,:] = (energy[jet,m]+energy[jet,:connections])**2-torch.sum((p[jet,m,:]+p[jet,:connections,:])**2,axis=1)
#                 edge_tensor[jet,m,m] = 0.
#                 edge_tensor[jet,m,m]=((energy[jet,m]+energy[jet,m])**2-torch.sum((p[jet,m,:]+p[jet,m,:])**2,axis=0))
                # inds_edge = torch.sqrt( (phi[jet,m]-phi[jet,:])**2 + (rapidity[jet,m]-rapidity[jet,:])**2 ).argsort()[:connections]
                # edge_tensor[jet,m,:] = torch.sqrt( (phi[jet,m]-phi[jet,inds_edge])**2 + (rapidity[jet,m]-rapidity[jet,inds_edge])**2 )
                edge_tensor[jet,m,:] = torch.sqrt( (phi[jet,m]-phi[jet,:connections])**2 + (rapidity[jet,m]-rapidity[jet,:connections])**2 )
#                 inds_edge = np.argsort( (energy[jet,m]+energy[jet,:stop_indx])**2-torch.sum((p[jet,m,:stop_indx]+p[jet,:stop_indx,:])**2,axis=1) )[:connections]
#                 edge_tensor[jet,m,inds_edge] = (energy[jet,m]+energy[jet,inds_edge])**2-torch.sum((p[jet,m,:]+p[jet,inds_edge,:])**2,axis=1)
            edges_exist_at = torch.where(edge_tensor[jet,:,:].abs()>0)
            
#             edge_indx_tensor[jet,:,:(edge_tensor[jet,:,:].abs()>0).sum()] = fixed_edges_list 
            edge_indx_tensor[jet,:,:(edge_tensor[jet,:,:].abs()>0).sum()] = torch.cat((edges_exist_at[0][None,:],edges_exist_at[1][None,:]),dim=0).reshape((2,edges_exist_at[0].shape[0]))   
            edge_attr_matrix[jet,:(edge_tensor[jet,:,:].abs()>0).sum(),0]  =  edge_tensor[jet,edges_exist_at[0],edges_exist_at[1]].flatten()

    end_edges_indx_per_jet = (edge_attr_matrix!=0).sum(axis=1).int()
    keep_inds =  torch.where(end_edges_indx_per_jet>=connections)[0]
    
    edge_tensor = edge_tensor/edge_tensor.max()
    edge_attr_matrix = edge_attr_matrix/edge_attr_matrix.max()
    
    graph_help = torch.cat( ( (energy.max(axis=1).values/e_per_jet).reshape(x_data[:,0,3].shape[0],1),
                              (mass_per_jet).reshape(x_data[:,0,3].shape[0],1),
                              (end_multiplicity_indx_per_jet).reshape(x_data[:,0,3].shape[0],1).int(),
                              (end_edges_indx_per_jet).reshape(x_data[:,0,3].shape[0],1).int() ), dim=1)
    
    
    print()
    print('--- Calculating Quantum Hamiltonians ---')
    quantum_hamiltonians = torch.zeros((N,2,int(2**M),int(2**M) )) #0: coupling, 1: transverse
    for idx in tqdm(keep_inds):
        quantum_hamiltonians[idx,0,:,:] = create_hamiltonian_matrix_interaction(nodes_per_graph,
                                                                                  edge_indx_tensor[idx], 
                                                                                  edge_attr_matrix[idx])
        quantum_hamiltonians[idx,1,:,:] = create_hamiltonian_matrix_transverse(nodes_per_graph)
    
    
        
    return x_data[keep_inds,:nodes_per_graph], y_data[keep_inds].long(), edge_tensor[keep_inds], edge_indx_tensor[keep_inds].long(), edge_attr_matrix[keep_inds], graph_help[keep_inds], quantum_hamiltonians













def create_hamiltonian_matrix_interaction(nodes_per_graph,
                                          edge_indx_tensor, 
                                          edge_attr_matrix,
                                          term_dictionary =  {0: [np.identity(2),np.identity(2)], 
                                                              1: [qml.matrix(qml.PauliZ)(0), qml.matrix(qml.PauliZ)(0)]} ):
    # n_qubits == number of nodes in graph
    # graph == Data frame
    # weights == edge_attr
    n_qubits = nodes_per_graph # graph.x.shape[0]
    full_matrix = np.zeros((2 ** n_qubits, 2 ** n_qubits))
    # Creates the interaction component of the Hamiltonian
    weights = edge_attr_matrix #graph.edge_attr
    fraction = 1./4.
    number_of_terms = len(term_dictionary)
    for term in range(number_of_terms):
        for i, edge in enumerate(edge_indx_tensor.T): #graph.edge_index.T):
            interaction_term = 1
            for qubit in range(n_qubits):
                if qubit in edge:
                    if qubit==edge[0]:
                        interaction_term = np.kron(interaction_term, term_dictionary[term][0] )
                    elif qubit==edge[1]:
                        interaction_term = np.kron(interaction_term, term_dictionary[term][1] )
                else:
                    interaction_term = np.kron(interaction_term, np.identity(2) )
                    
        if np.all(term_dictionary[term][0]==qml.matrix(qml.PauliZ)(0)):
            full_matrix += -fraction*weights[i].numpy()*interaction_term
        else:
            full_matrix +=  fraction*weights[i].numpy()*interaction_term
    return torch.tensor(full_matrix)

def create_hamiltonian_matrix_transverse(nodes_per_graph):
    n_qubits = nodes_per_graph #graph.x.shape[0]
    full_matrix = np.zeros((2 ** n_qubits, 2 ** n_qubits))
    # Creates the bias components of the matrix
    for i in range(n_qubits):
        x_term = 1
        for j in range(n_qubits):
            if j==i:
                x_term = np.kron(x_term, qml.matrix(qml.PauliX)(0))
            else:
                x_term = np.kron(x_term, np.identity(2) )
        full_matrix += x_term
    return torch.tensor(full_matrix)