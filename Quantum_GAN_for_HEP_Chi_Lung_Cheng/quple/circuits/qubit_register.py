from cirq import GridQubit

from typing import (Any, Callable, cast, Dict, FrozenSet, Iterable, Iterator,
                    List, Optional, overload, Sequence, Set, Tuple, Type,
                    TYPE_CHECKING, TypeVar, Union)    

class QubitRegister():
    """Qubit Register (the quantum analog of a classical processor register)
    
    A qubit register keeps track of the qubits used in a quantum circuit. 
    
    Examples:
    --------
    >>> qr = quple.QubitRegister(5)
    >>> qr.size
    5
    >>> qr.qubits
    [cirq.GridQubit(0, 0),
     cirq.GridQubit(0, 1),
     cirq.GridQubit(0, 2),
     cirq.GridQubit(0, 3),
     cirq.GridQubit(0, 4)]
    >>> qubits = [cirq.GridQubit(1,2), cirq.GridQubit(1,3), cirq.GridQubit(2,2)]
    >>> qr.size
    3
    >>> qr = quple.QubitRegister(qubits)
    [cirq.GridQubit(1, 2), cirq.GridQubit(1, 3), cirq.GridQubit(2, 2)]
    """
    def __init__(self, qubits:Union[int, Sequence[GridQubit]]=0):
        """
        Args:
            qubits: If an integer is provided, it specifies the number of qubits 
                    for the qubit register with a linear qubit layout.
                    If a sequence of cirq Qubit objects is provided, it specifies
                    the qubits in the register.
        """
        if isinstance(qubits, int):
            self._size = qubits
            self._qubits = GridQubit.rect(1, qubits)
        else:
            self._size = len(qubits)
            self._qubits = list(qubits)

        
    def __getitem__(self, key: int) -> GridQubit:
        return self._qubits[key]
    
    @staticmethod
    def _is_unique_qubit_set(qubits: Union[List[Tuple[GridQubit]],Tuple[GridQubit]]):
        """Checks that qubits in a tuple or a list of tuples are unique
        
        Args:
            qubits: a tuple or a list of tuples of qubits for checking uniqueness
        Returns:
            True if the tuple or list of tuples of qubits are unique and False otherwise
        """
        if isinstance(qubits, tuple):
            return len(set(qubits)) == len(qubits)
        elif isinstance(qubits, list):
            return all(len(set(sub_qubits)) == len(sub_qubits) for sub_qubits in qubits)
        return False
    
    @staticmethod    
    def _parse_qubit_expression(qubit_expr, target_qubits):
        """Parse a qubit expression by the indices of some target qubits
        
        Args:
            qubit_expr: The qubit expression to parse. 
                        If integer, it specifies the index of the target qubit.
                        If range, slice or list of integers, it specifies the 
                        sequence of indices of the target qubits.
                        If tuple or list of tuples of integers, it specifies the 
                        tuple or list of tuples of qubits indexed by the given integers.
            target_qubits: A list of qubits based on which the 
                           qubit expression is parsed
                           
        Returns:
            a qubit expression (list, tuples or list of tuples of qubits) 
        """
        resolved_qubits = None
        try:
            if isinstance(qubit_expr, GridQubit):
                resolved_qubits = qubit_expr
            elif isinstance(qubit_expr, (int, slice)):
                resolved_qubits = target_qubits[qubit_expr]
            elif isinstance(qubit_expr, (tuple, list)):
                resolved_qubits = type(qubit_expr)([QubitRegister._parse_qubit_expression(i, target_qubits) \
                                                    for i in qubit_expr])
            elif isinstance(qubit_expr, range):
                resolved_qubits = [target_qubits[i] for i in qubit_expr]
            else:
                raise ValueError('Unsupported qubit expression {} ({})'.format(qubit_expr, type(qubit_expr)))
        except IndexError:
                raise IndexError('Qubit index out of range.') from None
        except TypeError:
                raise IndexError('Qubit index must be an integer') from None
        from pdb import set_trace
        return resolved_qubits
    
    def get(self, qubit_expr) -> Union[List[GridQubit], List[Tuple[GridQubit]]]:
        """Returns the qubits in the qubit register given a qubit expression
        """
        return QubitRegister._parse_qubit_expression(qubit_expr, self._qubits)
    
    @property
    def size(self) -> int:
        """Returns the number of qubits in the qubit register
        """
        return self._size

    @property
    def qubits(self) -> List[GridQubit]:
        """Returns the qubits in the qubit register
        """
        return self._qubits    