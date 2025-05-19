from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from okey_ilp import OkeyILPSolver
import time

app = FastAPI(
    title="Okey Solver API",
    description="API for solving Okey hands using Integer Linear Programming",
    version="1.0.0"
)

class PieceData(BaseModel):
    color: str
    number: Optional[int]

class OkeyRequest(BaseModel):
    pieces: List[PieceData]
    okey_color: str
    okey_number: int

class MeldResponse(BaseModel):
    pieces: List[str]
    value: int

class OkeyResponse(BaseModel):
    melds: List[MeldResponse]
    total_score: int
    can_open: bool
    number_of_triples: int
    number_of_sides: int
    execution_time: float

@app.post("/solve", response_model=OkeyResponse)
async def solve_okey_hand(request: OkeyRequest):
    try:
        # Convert pieces to the format expected by OkeyILPSolver
        pieces_data = [{"color": p.color, "number": p.number} for p in request.pieces]
        
        # Start timing
        start_time = time.time()
        
        # Create solver instance and solve
        solver = OkeyILPSolver(
            pieces=pieces_data,
            okey_color=request.okey_color,
            okey_number=request.okey_number
        )
        
        # Get optimal melds and their ILP-calculated weights
        optimal_melds_with_weights = solver.solve() # Returns List[Tuple[List[Piece], int]]
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Process melds and calculate total score
        processed_melds = []
        total_score = 0
        
        # Each item is now a tuple: (list_of_piece_objects, ilp_calculated_weight)
        for piece_objects_in_meld, ilp_meld_weight in optimal_melds_with_weights:
            meld_pieces_str_list = [] 
            
            for piece in piece_objects_in_meld: # piece is Piece object
                if piece.type == 'okey':
                    piece_str = 'JOKER'
                else: # piece.type == 'piece' or 'fake_okey' (for string representation)
                    piece_str = f"{piece.color} {piece.number}"
                meld_pieces_str_list.append(piece_str)
            
            # The value is taken directly from the ILP solver.
            processed_melds.append(MeldResponse(
                pieces=meld_pieces_str_list,
                value=ilp_meld_weight # Use the weight from ILP
            ))
            total_score += ilp_meld_weight # Add this weight to total_score
        
        # Calculate if hand can open and number of triples/sides
        can_open = total_score >= 101
        number_of_triples = total_score // 3
        number_of_sides = total_score % 3
        
        return OkeyResponse(
            melds=processed_melds,
            total_score=total_score,
            can_open=can_open,
            number_of_triples=number_of_triples,
            number_of_sides=number_of_sides,
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Welcome to Okey Solver API",
        "endpoints": {
            "/solve": "POST - Solve an Okey hand",
            "/docs": "GET - API documentation"
        }
    } 