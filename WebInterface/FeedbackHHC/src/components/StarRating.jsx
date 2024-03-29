import { FaStar, FaRegStar, FaStarHalfAlt } from 'react-icons/fa';

const StarRating = ({ rating }) => {
  const totalStars = 5;
  let stars = [];

  for (let i = 1; i <= Math.floor(rating); i++) {
    stars.push(<FaStar key={i} color="#ffc107" style={{"font-size":"50px"}}/>);
  }

  if (rating % 1 !== 0) {
    stars.push(<FaStarHalfAlt key="half" color="#ffc107" style={{"font-size":"50px"}}/>);
  }

  for (let i = Math.ceil(rating) + 1; i <= totalStars; i++) {
    stars.push(<FaRegStar key={i} color="#ffc107" style={{"font-size":"50px"}}/>);
  }

  return <div>{stars}</div>;
};

export default StarRating;
